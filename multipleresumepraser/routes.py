from fastapi import APIRouter, HTTPException, Query
import os
from fastapi import File, UploadFile, HTTPException, Request
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import List, Optional
from .text_extraction import extract_and_clean_text, clean_text
from .main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import Vectorizer
from core.custom_logger import CustomLogger

# Initialize your parser with default provider from config
parser = ResumeParser()
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
# Initialize database operations
skills_ops = SkillsTitlesOperations(skills_titles_collection)
vectorizer = Vectorizer()
resume_ops = ResumeOperations(collection, vectorizer)

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

# Configure logging
logger_manager = CustomLogger()
logging = logger_manager.get_logger("groqcloud_llm")


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


def normalize_text_data(text: str) -> str:
    """
    Normalize text data by converting to lowercase and removing extra spaces.

    Args:
        text (str): The input text to normalize

    Returns:
        str: Normalized text (lowercase, trimmed spaces)
    """
    if not text or not isinstance(text, str):
        return text
    return text.strip().lower()


def normalize_text_list(text_list: List[str]) -> List[str]:
    """
    Normalize a list of text strings by converting to lowercase and removing extra spaces.

    Args:
        text_list (List[str]): List of text strings to normalize

    Returns:
        List[str]: List of normalized text strings
    """
    if not text_list or not isinstance(text_list, list):
        return text_list
    return [normalize_text_data(text) for text in text_list if text]


@router.post("/switch-provider/")
async def switch_llm_provider(
    provider: str = Query(..., description="LLM provider to use ('groq' or 'ollama')"),
    api_keys: Optional[List[str]] = Query(
        None, description="Groq API keys (only for Groq provider)"
    ),
):
    """
    Switch between LLM providers (Groq Cloud or Ollama).
    """
    global parser

    try:
        if provider.lower() not in ["groq", "ollama"]:
            raise HTTPException(
                status_code=400, detail="Invalid provider. Must be 'groq' or 'ollama'"
            )

        # Switch the parser to new provider
        parser.switch_provider(provider.lower(), api_keys)

        current_provider = "Ollama" if parser.use_ollama else "Groq Cloud"

        logging.info(f"Switched to {current_provider} provider")

        return {
            "message": f"Successfully switched to {current_provider}",
            "provider": current_provider,
            "status": "success",
        }

    except Exception as e:
        logging.error(f"Failed to switch provider: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to switch provider: {str(e)}"
        )


@router.get("/provider-info/")
async def get_provider_info():
    """
    Get current LLM provider information.
    """
    try:
        current_provider = "Ollama" if parser.use_ollama else "Groq Cloud"

        info = {
            "current_provider": current_provider,
            "provider_type": "local" if parser.use_ollama else "api",
        }

        if parser.use_ollama:
            info.update(
                {
                    "model": parser.ollama_config.primary_model,
                    "api_url": parser.ollama_config.api_url,
                    "available_models": parser._get_available_ollama_models(),
                }
            )
        else:
            info.update(
                {
                    "model": parser.groq_config.primary_model,
                    "api_keys_count": len(parser.api_keys),
                    "current_key_index": parser.current_key_index,
                    "api_usage": parser.api_usage,
                }
            )

        return info

    except Exception as e:
        logging.error(f"Failed to get provider info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get provider info: {str(e)}"
        )


@router.post("/grouqcloud/")
async def extract_clean_text_llam3_3b(
    file: UploadFile = File(...),
    provider: Optional[str] = Query(
        None, description="LLM provider to use ('groq' or 'ollama')"
    ),
):
    """
    Endpoint to extract and clean text from uploaded file for multiple resume parser.
    Optionally specify which LLM provider to use for this request.
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

        cleaned_text = extract_and_clean_text(file_location)
        print(cleaned_text)
        resume_parser = parser.process_resume(cleaned_text)
        print(resume_parser)
        # Check if the resume_parser is empty or not

        resume_parser = json.dumps(resume_parser, indent=4)
        # print(resume_parser)
        print(type(resume_parser))

        resume_parser = json.loads(resume_parser)
        print(resume_parser)
        print(type(resume_parser))

        if not resume_parser:
            raise HTTPException(status_code=400, detail="No resume data found.")

        # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0  # Calculate experience
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        # Normalize skills and may_also_known_skills
        if "skills" in resume_parser and resume_parser["skills"]:
            resume_parser["skills"] = normalize_text_list(resume_parser["skills"])

        if (
            "may_also_known_skills" in resume_parser
            and resume_parser["may_also_known_skills"]
        ):
            resume_parser["may_also_known_skills"] = normalize_text_list(
                resume_parser["may_also_known_skills"]
            )

        # Normalize job titles in experience
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience and experience["title"]:
                    experience["title"] = normalize_text_data(experience["title"])

        # Fix the experience titles extraction (already normalized)
        experience_titles = []
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience:
                    experience_titles.append(
                        experience["title"]
                    )  # Fix the skills extraction (already normalized)
        skills = []
        if "skills" in resume_parser:
            skills = resume_parser["skills"]

        # Ensure skills and experience_titles are normalized before adding to skills_titles collection
        normalized_skills = normalize_text_list(skills) if skills else []
        normalized_experience_titles = (
            normalize_text_list(experience_titles) if experience_titles else []
        )

        resume_ops.create_resume(resume_parser)
        skills_ops.add_multiple_skills(normalized_skills)
        skills_ops.add_multiple_titles(normalized_experience_titles)

        logging.info(f"Added normalized skills: {normalized_skills}")
        logging.info(
            f"Added normalized experience titles: {normalized_experience_titles}"
        )

        # Delete the temporary file
        os.remove(file_location)

        return {
            "filename": file.filename,
            "cleaned_text": cleaned_text,
            "resume_parser": resume_parser,
            # "resume_parser_exp": resume_parser_exp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel
from typing import List, Dict, Any


class ResumeData(BaseModel):
    text: str
    # Add other fields as necessary


@router.post("/grouqcloud-text/")
async def extract_clean_text_from_raw(request: ResumeData):
    """
    Endpoint to extract and clean text from raw resume text input for LLM model.
    """
    try:

        resume_text = request.text

        if not resume_text:
            raise HTTPException(status_code=400, detail="No resume text provided.")

        # Clean the input text
        cleaned_text = clean_text(resume_text)

        # Parse the cleaned resume
        resume_parser = parser.process_resume(cleaned_text)

        if not resume_parser:
            raise HTTPException(status_code=400, detail="No resume data found.")

        # Convert to dict if needed (already is in dict if process_resume returns it)
        if isinstance(resume_parser, str):
            resume_parser = json.loads(resume_parser)

        # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0  # Calculate experience
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        # Normalize skills and may_also_known_skills
        if "skills" in resume_parser and resume_parser["skills"]:
            resume_parser["skills"] = normalize_text_list(resume_parser["skills"])

        if (
            "may_also_known_skills" in resume_parser
            and resume_parser["may_also_known_skills"]
        ):
            resume_parser["may_also_known_skills"] = normalize_text_list(
                resume_parser["may_also_known_skills"]
            )

        # Normalize job titles in experience
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience and experience["title"]:
                    experience["title"] = normalize_text_data(experience["title"])

        # Extract experience titles (already normalized)
        experience_titles = []
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience:
                    experience_titles.append(
                        experience["title"]
                    )  # Extract skills (already normalized)
        skills = []
        if "skills" in resume_parser:
            skills = resume_parser["skills"]

        # Ensure skills and experience_titles are normalized before adding to skills_titles collection
        normalized_skills = normalize_text_list(skills) if skills else []
        normalized_experience_titles = (
            normalize_text_list(experience_titles) if experience_titles else []
        )  # Store in database
        resume_ops.create_resume(resume_parser)
        skills_ops.add_multiple_skills(normalized_skills)
        skills_ops.add_multiple_titles(normalized_experience_titles)

        logging.info(f"Added normalized skills: {normalized_skills}")
        logging.info(
            f"Added normalized experience titles: {normalized_experience_titles}"
        )

        return {
            "cleaned_text": cleaned_text,
            "resume_parser": resume_parser,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
