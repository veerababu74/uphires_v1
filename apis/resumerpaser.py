from fastapi import APIRouter, HTTPException
import os
from fastapi import File, UploadFile, HTTPException
from pathlib import Path
from datetime import datetime, timedelta, date
from GroqcloudLLM.text_extraction import extract_and_clean_text
from GroqcloudLLM.main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from core.custom_logger import CustomLogger
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
from typing import List, Dict, Any, Optional


import json
import re


def clean_and_extract_skills(skills_input: List[str]) -> List[str]:
    """
    Extract and clean individual skills from various input formats

    Handles scenarios like:
    - "python,machine learning,"
    - "technical skills:python,sql,opencv,azure cloud"
    - "Python | React | AWS"
    - "Skills: Java, Spring Boot, Microservices"
    - Mixed formats in single list
    """

    if not skills_input:
        return []

    all_skills = []

    for skill_item in skills_input:
        if not skill_item or not isinstance(skill_item, str):
            continue

        # Convert to lowercase for processing
        skill_text = skill_item.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            r"^technical\s*skills?\s*:?\s*",
            r"^skills?\s*:?\s*",
            r"^technologies?\s*:?\s*",
            r"^expertise\s*:?\s*",
            r"^programming\s*:?\s*",
            r"^tools?\s*:?\s*",
        ]

        for prefix in prefixes_to_remove:
            skill_text = re.sub(prefix, "", skill_text, flags=re.IGNORECASE)

        # Split by various delimiters
        delimiters = [",", "|", ";", "/", "\\n", "\\r", "\n", "\r"]

        # Create regex pattern for splitting
        delimiter_pattern = "|".join(map(re.escape, delimiters))
        skills_parts = re.split(delimiter_pattern, skill_text)

        # Process each skill part
        for skill in skills_parts:
            cleaned_skill = clean_individual_skill(skill)
            if cleaned_skill:
                all_skills.append(cleaned_skill)

    # Remove duplicates while preserving order
    unique_skills = []
    seen = set()
    for skill in all_skills:
        skill_lower = skill.lower()
        if skill_lower not in seen:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills


def clean_individual_skill(skill: str) -> str:
    """Clean and normalize individual skill"""
    if not skill:
        return ""

    # Remove extra whitespace and common unwanted characters
    skill = skill.strip(" \t\n\r,;|/")

    # Remove numbers at the beginning (like "1. Python")
    skill = re.sub(r"^\d+\.?\s*", "", skill)

    # Remove brackets and parentheses content
    skill = re.sub(r"\([^)]*\)", "", skill)
    skill = re.sub(r"\[[^\]]*\]", "", skill)

    # Remove extra spaces
    skill = re.sub(r"\s+", " ", skill).strip()

    # Skip if too short or contains only special characters
    if len(skill) < 2 or re.match(r"^[^a-zA-Z0-9]*$", skill):
        return ""

    return skill


def process_user_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process the entire user JSON and clean skills"""

    # Create a copy to avoid modifying original
    processed_data = json_data.copy()

    # Process main skills
    if "skills" in processed_data:
        processed_data["skills"] = clean_and_extract_skills(processed_data["skills"])

    # Process may_also_known_skills
    if "may_also_known_skills" in processed_data:
        processed_data["may_also_known_skills"] = clean_and_extract_skills(
            processed_data["may_also_known_skills"]
        )

    return processed_data


# Initialize your parser with API keys (replace with your actual keys)
parser = ResumeParser()
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
# Initialize database operations
skills_ops = SkillsTitlesOperations(skills_titles_collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)

# Create a router instance
router = APIRouter()

BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


logger_manager = CustomLogger()
logging = logger_manager.get_logger("add_userdata")


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """
    Cleanup temporary directory by deleting files older than the specified age limit.
    """
    now = datetime.now()
    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")


@router.post("/resume-parser", tags=["Resume Parser", "final_apis"])
async def upload_resume(file: UploadFile = File(...)):
    """
    Endpoint to extract and clean text from uploaded file for llm model.
    """
    try:
        # Define the path to save the uploaded file
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        # Save the uploaded file
        with open(file_location, "wb") as temp_file:
            temp_file.write(await file.read())

        # Extract text
        _, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
            )

        # Extract and save the total resume text
        total_resume_text = extract_and_clean_text(file_location)
        print("\nTotal Resume Text:")
        print("=" * 50)
        print(total_resume_text)
        print("=" * 50)

        # Process the resume
        resume_parser = parser.process_resume(total_resume_text)

        if not resume_parser:
            raise HTTPException(
                status_code=400, detail="No resume data found."
            )  # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0

        # Calculate experience
        # res = calculator.calculate_experience(resume_parser)
        # resume_parser["total_experience"] = format_experience(res[0], res[1])

        resume_parser = process_user_json(resume_parser)

        return {
            "filename": file.filename,
            "total_resume_text": total_resume_text,
            "resume_parser": resume_parser,
        }
    except HTTPException as http_ex:
        # Re-raise HTTPException as-is (don't modify it)
        logging.info(f"HTTPException raised in upload: {http_ex.detail}")
        raise http_ex

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(f"Error in upload_resume: {error_details}")

        detail_message = f"Error processing file: {error_details['error_type']}: {error_details['error_message']}"
        if not error_details["error_message"]:
            detail_message = f"Error processing file: {error_details['error_type']} - Check logs for details"

        raise HTTPException(status_code=500, detail=detail_message)
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


def process_single_resume(
    file_content: bytes, filename: str, file_id: str
) -> Dict[str, Any]:
    """
    Process a single resume file and return the result.
    This function is designed to be thread-safe.
    """
    file_location = None
    try:
        # Create unique file path for this thread
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"{file_id}_{timestamp}_{filename}"
        file_location = os.path.join(TEMP_FOLDER, safe_filename)

        # Save the file
        with open(file_location, "wb") as temp_file:
            temp_file.write(file_content)

        # Check file extension
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
            return {
                "filename": filename,
                "success": False,
                "error": f"Unsupported file type: {file_extension}. Only .txt, .pdf, and .docx are supported.",
                "error_type": "UNSUPPORTED_FILE_TYPE",
            }

        # Extract text
        total_resume_text = extract_and_clean_text(file_location)

        # Process the resume with parser
        resume_parser = parser.process_resume(total_resume_text)

        if not resume_parser:
            return {
                "filename": filename,
                "success": False,
                "error": "No resume data found in the file.",
                "error_type": "NO_DATA_FOUND",
            }

        # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0

        # Process and clean the JSON data
        resume_parser = process_user_json(resume_parser)

        return {
            "filename": filename,
            "success": True,
            "total_resume_text": total_resume_text,
            "resume_parser": resume_parser,
        }

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(f"Error processing {filename}: {error_details}")

        return {
            "filename": filename,
            "success": False,
            "error": f"{error_details['error_type']}: {error_details['error_message']}",
            "error_type": error_details["error_type"],
        }

    finally:
        # Clean up the temporary file
        if file_location and os.path.exists(file_location):
            try:
                os.remove(file_location)
            except Exception as cleanup_error:
                logging.warning(
                    f"Failed to cleanup file {file_location}: {cleanup_error}"
                )


@router.post("/resume-parser-multiple", tags=["Resume Parser", "final_apis"])
async def upload_multiple_resumes(files: List[UploadFile] = File(...)):
    """
    Endpoint to extract and clean text from multiple uploaded files using threading for better performance.

    Args:
        files: List of uploaded files (supports .txt, .pdf, .docx)

    Returns:
        Dict containing:
        - total_files: Total number of files processed
        - successful_files: Number of successfully processed files
        - failed_files: Number of files that failed to process
        - results: List of processing results for each file
        - processing_time: Total time taken for processing
    """
    start_time = datetime.now()

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 50:  # Limit to prevent resource exhaustion
            raise HTTPException(
                status_code=400, detail="Maximum 50 files allowed per request"
            )

        # Read all files into memory first
        file_data = []
        for i, file in enumerate(files):
            if not file.filename:
                continue
            content = await file.read()
            file_data.append(
                {"content": content, "filename": file.filename, "file_id": f"file_{i}"}
            )

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files found")

        results = []
        successful_count = 0
        failed_count = 0

        # Use ThreadPoolExecutor for concurrent processing
        max_workers = min(len(file_data), 10)  # Limit concurrent threads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_single_resume,
                    file_info["content"],
                    file_info["filename"],
                    file_info["file_id"],
                ): file_info
                for file_info in file_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    file_info = future_to_file[future]
                    error_result = {
                        "filename": file_info["filename"],
                        "success": False,
                        "error": f"Threading error: {str(e)}",
                        "error_type": "THREADING_ERROR",
                    }
                    results.append(error_result)
                    failed_count += 1
                    logging.error(f"Threading error for {file_info['filename']}: {e}")

        # Sort results by filename for consistent output
        results.sort(key=lambda x: x["filename"])

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return {
            "total_files": len(file_data),
            "successful_files": successful_count,
            "failed_files": failed_count,
            "processing_time_seconds": round(processing_time, 2),
            "results": results,
            "summary": {
                "success_rate": (
                    round((successful_count / len(file_data)) * 100, 2)
                    if file_data
                    else 0
                ),
                "avg_time_per_file": (
                    round(processing_time / len(file_data), 2) if file_data else 0
                ),
            },
        }

    except HTTPException as http_ex:
        logging.info(f"HTTPException in upload_multiple_resumes: {http_ex.detail}")
        raise http_ex

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(f"Error in upload_multiple_resumes: {error_details}")

        detail_message = f"Error processing multiple files: {error_details['error_type']}: {error_details['error_message']}"
        raise HTTPException(status_code=500, detail=detail_message)


def process_resume_for_multiprocessing(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single resume file for multiprocessing.
    This function is designed to be pickle-able for multiprocessing.
    """
    try:
        # Initialize parser for this process
        local_parser = ResumeParser()

        filename = file_data["filename"]
        content = file_data["content"]
        file_id = file_data["file_id"]

        # Create unique file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"{file_id}_{timestamp}_{filename}"
        file_location = os.path.join(TEMP_FOLDER, safe_filename)

        # Save the file
        with open(file_location, "wb") as temp_file:
            temp_file.write(content)

        try:
            # Check file extension
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                return {
                    "filename": filename,
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}. Only .txt, .pdf, and .docx are supported.",
                    "error_type": "UNSUPPORTED_FILE_TYPE",
                }

            # Extract text
            total_resume_text = extract_and_clean_text(file_location)

            # Process the resume with parser
            resume_parser = local_parser.process_resume(total_resume_text)

            if not resume_parser:
                return {
                    "filename": filename,
                    "success": False,
                    "error": "No resume data found in the file.",
                    "error_type": "NO_DATA_FOUND",
                }

            # Initialize total_experience if not present
            if "total_experience" not in resume_parser:
                resume_parser["total_experience"] = 0

            # Process and clean the JSON data
            resume_parser = process_user_json(resume_parser)

            return {
                "filename": filename,
                "success": True,
                "total_resume_text": total_resume_text,
                "resume_parser": resume_parser,
            }

        finally:
            # Clean up the temporary file
            if os.path.exists(file_location):
                try:
                    os.remove(file_location)
                except Exception as cleanup_error:
                    pass  # Ignore cleanup errors in multiprocessing

    except Exception as e:
        return {
            "filename": file_data.get("filename", "unknown"),
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "error_type": type(e).__name__,
        }


@router.post("/resume-parser-multiple-mp", tags=["Resume Parser", "final_apis"])
async def upload_multiple_resumes_multiprocessing(files: List[UploadFile] = File(...)):
    """
    Endpoint to extract and clean text from multiple uploaded files using multiprocessing for maximum performance.
    Best for CPU-intensive processing with many files.

    Args:
        files: List of uploaded files (supports .txt, .pdf, .docx)

    Returns:
        Dict containing:
        - total_files: Total number of files processed
        - successful_files: Number of successfully processed files
        - failed_files: Number of files that failed to process
        - results: List of processing results for each file
        - processing_time: Total time taken for processing
    """
    start_time = datetime.now()

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 100:  # Higher limit for multiprocessing
            raise HTTPException(
                status_code=400, detail="Maximum 100 files allowed per request"
            )

        # Read all files into memory first
        file_data = []
        for i, file in enumerate(files):
            if not file.filename:
                continue
            content = await file.read()
            file_data.append(
                {"content": content, "filename": file.filename, "file_id": f"file_{i}"}
            )

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files found")

        results = []
        successful_count = 0
        failed_count = 0

        # Use ProcessPoolExecutor for CPU-bound tasks
        max_workers = min(len(file_data), multiprocessing.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_resume_for_multiprocessing, file_info
                ): file_info
                for file_info in file_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    results.append(result)

                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    file_info = future_to_file[future]
                    error_result = {
                        "filename": file_info["filename"],
                        "success": False,
                        "error": f"Multiprocessing error: {str(e)}",
                        "error_type": "MULTIPROCESSING_ERROR",
                    }
                    results.append(error_result)
                    failed_count += 1

        # Sort results by filename for consistent output
        results.sort(key=lambda x: x["filename"])

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return {
            "total_files": len(file_data),
            "successful_files": successful_count,
            "failed_files": failed_count,
            "processing_time_seconds": round(processing_time, 2),
            "processing_method": "multiprocessing",
            "workers_used": max_workers,
            "results": results,
            "summary": {
                "success_rate": (
                    round((successful_count / len(file_data)) * 100, 2)
                    if file_data
                    else 0
                ),
                "avg_time_per_file": (
                    round(processing_time / len(file_data), 2) if file_data else 0
                ),
                "performance_boost": f"Used {max_workers} CPU cores for parallel processing",
            },
        }

    except HTTPException as http_ex:
        logging.info(
            f"HTTPException in upload_multiple_resumes_multiprocessing: {http_ex.detail}"
        )
        raise http_ex

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(
            f"Error in upload_multiple_resumes_multiprocessing: {error_details}"
        )

        detail_message = f"Error processing multiple files with multiprocessing: {error_details['error_type']}: {error_details['error_message']}"
        raise HTTPException(status_code=500, detail=detail_message)


@router.get("/resume-parser-info", tags=["Resume Parser"])
async def get_resume_parser_info():
    """
    Get information about available resume parsing endpoints and their recommended use cases.
    """
    return {
        "available_endpoints": {
            "single_file": {
                "endpoint": "/resume-parser",
                "method": "POST",
                "description": "Process a single resume file",
                "use_case": "Single file processing, testing, or small scale operations",
                "max_files": 1,
            },
            "multiple_files_threading": {
                "endpoint": "/resume-parser-multiple",
                "method": "POST",
                "description": "Process multiple resume files using threading",
                "use_case": "I/O bound operations, moderate file counts (1-50 files)",
                "max_files": 50,
                "performance": "Good for I/O bound tasks like file reading and API calls",
            },
            "multiple_files_multiprocessing": {
                "endpoint": "/resume-parser-multiple-mp",
                "method": "POST",
                "description": "Process multiple resume files using multiprocessing",
                "use_case": "CPU-intensive operations, large file counts (1-100 files)",
                "max_files": 100,
                "performance": "Best for CPU-bound tasks like text processing and parsing",
            },
        },
        "recommendations": {
            "1-5_files": "Use /resume-parser or /resume-parser-multiple",
            "5-20_files": "Use /resume-parser-multiple (threading)",
            "20+_files": "Use /resume-parser-multiple-mp (multiprocessing)",
            "cpu_intensive": "Use /resume-parser-multiple-mp",
            "io_intensive": "Use /resume-parser-multiple",
        },
        "supported_formats": [".txt", ".pdf", ".docx"],
        "system_info": {
            "cpu_cores": multiprocessing.cpu_count(),
            "temp_directory": TEMP_FOLDER,
        },
    }
