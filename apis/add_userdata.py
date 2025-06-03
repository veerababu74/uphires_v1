from fastapi import APIRouter, HTTPException, status
import os
from fastapi import File, UploadFile, HTTPException
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from GroqcloudLLM.main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from core.custom_logger import CustomLogger
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional

# Initialize your parser with API keys (replace with your actual keys)
parser = ResumeParser()
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
# Initialize database operations
skills_ops = SkillsTitlesOperations(skills_titles_collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)

# Create a router instance
router = APIRouter(prefix="/add_user", tags=["add_user"])

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


def is_empty_or_whitespace(value) -> bool:
    """
    Check if a value is None, empty string, or contains only whitespace.

    Args:
        value: The value to check

    Returns:
        bool: True if the value is empty or whitespace-only
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def filter_empty_strings_from_list(text_list: List[str]) -> List[str]:
    """
    Filter out empty strings and whitespace-only strings from a list.

    Args:
        text_list (List[str]): List of text strings to filter

    Returns:
        List[str]: List with empty/whitespace strings removed
    """
    if not text_list or not isinstance(text_list, list):
        return []
    return [
        text.strip()
        for text in text_list
        if text and isinstance(text, str) and text.strip()
    ]


def clean_string_field(value: Optional[str]) -> Optional[str]:
    """
    Clean a string field by returning None if it's empty or whitespace-only.

    Args:
        value: The string value to clean

    Returns:
        Optional[str]: Cleaned string or None if empty/whitespace
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return value


def clean_resume_data(resume_dict: dict) -> dict:
    """
    Clean all string fields in resume data by removing empty/whitespace-only values.

    Args:
        resume_dict (dict): The resume dictionary to clean

    Returns:
        dict: Cleaned resume dictionary
    """
    # Clean contact details
    if "contact_details" in resume_dict:
        contact = resume_dict["contact_details"]

        # Clean optional string fields in contact details
        contact["alternative_phone"] = clean_string_field(
            contact.get("alternative_phone")
        )
        contact["gender"] = clean_string_field(contact.get("gender"))
        contact["naukri_profile"] = clean_string_field(contact.get("naukri_profile"))
        contact["linkedin_profile"] = clean_string_field(
            contact.get("linkedin_profile")
        )
        contact["portfolio_link"] = clean_string_field(contact.get("portfolio_link"))
        contact["aadhar_card"] = clean_string_field(contact.get("aadhar_card"))

        # Clean required string fields (but keep them if they have content)
        contact["name"] = clean_string_field(contact.get("name")) or contact.get("name")
        contact["current_city"] = clean_string_field(
            contact.get("current_city")
        ) or contact.get("current_city")
        contact["pan_card"] = clean_string_field(
            contact.get("pan_card")
        ) or contact.get("pan_card")

        # Clean looking_for_jobs_in list
        if "looking_for_jobs_in" in contact:
            contact["looking_for_jobs_in"] = filter_empty_strings_from_list(
                contact["looking_for_jobs_in"]
            )

    # Clean main resume fields
    resume_dict["user_id"] = clean_string_field(
        resume_dict.get("user_id")
    ) or resume_dict.get("user_id")
    resume_dict["username"] = clean_string_field(
        resume_dict.get("username")
    ) or resume_dict.get("username")
    resume_dict["notice_period"] = clean_string_field(resume_dict.get("notice_period"))
    resume_dict["currency"] = clean_string_field(resume_dict.get("currency"))
    resume_dict["pay_duration"] = clean_string_field(resume_dict.get("pay_duration"))
    resume_dict["source"] = clean_string_field(resume_dict.get("source"))
    resume_dict["last_working_day"] = clean_string_field(
        resume_dict.get("last_working_day")
    )
    resume_dict["comment"] = clean_string_field(resume_dict.get("comment"))
    resume_dict["exit_reason"] = clean_string_field(resume_dict.get("exit_reason"))

    # Clean skills lists
    if "skills" in resume_dict:
        resume_dict["skills"] = filter_empty_strings_from_list(resume_dict["skills"])

    if "may_also_known_skills" in resume_dict:
        resume_dict["may_also_known_skills"] = filter_empty_strings_from_list(
            resume_dict["may_also_known_skills"]
        )

    if "labels" in resume_dict and resume_dict["labels"]:
        resume_dict["labels"] = filter_empty_strings_from_list(resume_dict["labels"])

    # Clean experience data
    if "experience" in resume_dict and resume_dict["experience"]:
        for exp in resume_dict["experience"]:
            exp["company"] = clean_string_field(exp.get("company")) or exp.get(
                "company"
            )
            exp["title"] = clean_string_field(exp.get("title")) or exp.get("title")
            exp["from_date"] = clean_string_field(exp.get("from_date")) or exp.get(
                "from_date"
            )
            exp["to"] = clean_string_field(exp.get("to"))
            if "until" in exp:
                exp["until"] = clean_string_field(exp.get("until"))

    # Clean academic details
    if "academic_details" in resume_dict and resume_dict["academic_details"]:
        for edu in resume_dict["academic_details"]:
            edu["education"] = clean_string_field(edu.get("education")) or edu.get(
                "education"
            )
            edu["college"] = clean_string_field(edu.get("college")) or edu.get(
                "college"
            )

    return resume_dict


@router.post("/submit-details")
async def add_user_resume_details(resume_data: ResumeData):
    """
    Submit resume details and save to MongoDB with proper vector embeddings
    """
    try:
        logging.info(
            "Starting submit_resume_details function"
        )  # Convert Pydantic model to dictionary
        resume_dict = resume_data.model_dump()
        resume_dict["created_at"] = datetime.now(timezone.utc)

        # Clean all empty/whitespace-only strings from the data
        resume_dict = clean_resume_data(resume_dict)

        logging.info(
            "Successfully converted and cleaned resume data"
        )  # Check for existing records with same PAN card, Aadhar card, mobile number, or email
        contact_details = resume_dict["contact_details"]

        # Create a query to check for existing records (only check non-empty values)
        existing_conditions = []

        if contact_details.get("pan_card"):
            existing_conditions.append(
                {"contact_details.pan_card": contact_details["pan_card"]}
            )

        if contact_details.get("aadhar_card"):
            existing_conditions.append(
                {"contact_details.aadhar_card": contact_details["aadhar_card"]}
            )

        if contact_details.get("phone"):
            existing_conditions.append(
                {"contact_details.phone": contact_details["phone"]}
            )

        if contact_details.get("email"):
            existing_conditions.append(
                {"contact_details.email": contact_details["email"]}
            )
        # Only create query if we have conditions to check
        if existing_conditions:
            existing_query = {"$or": existing_conditions}
        else:
            existing_query = None

        # Check if any matching records exist
        existing_record = None
        if existing_query:
            try:
                logging.info("Checking for existing records in database")
                existing_record = collection.find_one(existing_query)
                logging.info(
                    f"Database query completed. Found existing record: {existing_record is not None}"
                )
            except Exception as db_error:
                logging.error(f"Database query error: {str(db_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database connection error: {str(db_error)}",
                )

        if existing_record:
            # Determine which field(s) caused the conflict
            conflicts = []
            if (
                existing_record["contact_details"].get("pan_card")
                and contact_details.get("pan_card")
                and existing_record["contact_details"].get("pan_card")
                == contact_details["pan_card"]
            ):
                conflicts.append("PAN Card")
            if (
                existing_record["contact_details"].get("aadhar_card")
                and contact_details.get("aadhar_card")
                and existing_record["contact_details"].get("aadhar_card")
                == contact_details["aadhar_card"]
            ):
                conflicts.append("Aadhar Card")
            if (
                existing_record["contact_details"].get("phone")
                and contact_details.get("phone")
                and existing_record["contact_details"].get("phone")
                == contact_details["phone"]
            ):
                conflicts.append("Mobile Number")
            if (
                existing_record["contact_details"].get("email")
                and contact_details.get("email")
                and existing_record["contact_details"].get("email")
                == contact_details["email"]
            ):
                conflicts.append("Email")

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Record already exists with the same {', '.join(conflicts)}",
            )  # Handle experience dates - they're already strings based on the model
        if "experience" in resume_dict and resume_dict["experience"]:
            for exp in resume_dict["experience"]:
                # Ensure the field names match between model and processing
                if "to" in exp and exp["to"] is not None:
                    exp["until"] = exp["to"]
                    del exp["to"]
                # Normalize job titles (only if not empty)
                if "title" in exp and exp["title"]:
                    exp["title"] = normalize_text_data(exp["title"])

        # Convert contact details URLs to strings (only if not empty/None)
        contact_details = resume_dict["contact_details"]
        if contact_details.get("naukri_profile"):
            contact_details["naukri_profile"] = str(contact_details["naukri_profile"])
        if contact_details.get("linkedin_profile"):
            contact_details["linkedin_profile"] = str(
                contact_details["linkedin_profile"]
            )
        if contact_details.get("portfolio_link"):
            contact_details["portfolio_link"] = str(contact_details["portfolio_link"])

        # Normalize skills and may_also_known_skills (already cleaned of empty strings)
        if "skills" in resume_dict and resume_dict["skills"]:
            resume_dict["skills"] = normalize_text_list(resume_dict["skills"])

        if (
            "may_also_known_skills" in resume_dict
            and resume_dict["may_also_known_skills"]
        ):
            resume_dict["may_also_known_skills"] = normalize_text_list(
                resume_dict["may_also_known_skills"]
            )

        # last_working_day is already a string based on the model definition
        # No conversion needed        # Combine all data into one proper resume format for text representation
        combined_resume = f"""
RESUME

PERSONAL INFORMATION
-------------------
Name: {contact_details.get('name', 'N/A')}
Contact Details:
  Email: {contact_details.get('email', 'N/A')}
  Phone: {contact_details.get('phone', 'N/A')}
  Alternative Phone: {contact_details.get('alternative_phone', 'N/A')}
  Current City: {contact_details.get('current_city', 'N/A')}
  Looking for jobs in: {', '.join(contact_details.get('looking_for_jobs_in', [])) if contact_details.get('looking_for_jobs_in') else 'N/A'}
  Gender: {contact_details.get('gender', 'N/A')}
  Age: {contact_details.get('age', 'N/A')}
  PAN Card: {contact_details.get('pan_card', 'N/A')}
  Aadhar Card: {contact_details.get('aadhar_card', 'N/A')}

PROFESSIONAL SUMMARY
-------------------
Total Experience: {resume_dict.get('total_experience', 'N/A')} years
Notice Period: {resume_dict.get('notice_period', 'N/A')} days
Current Salary: {resume_dict.get('currency', 'N/A')} {resume_dict.get('current_salary', 'N/A')} ({resume_dict.get('pay_duration', 'N/A')})
Expected Salary: {resume_dict.get('currency', 'N/A')} {resume_dict.get('expected_salary', 'N/A')} ({resume_dict.get('pay_duration', 'N/A')})
Hike Expected: {resume_dict.get('hike', 'N/A')}%
Last Working Day: {resume_dict.get('last_working_day', 'N/A')}
Exit Reason: {resume_dict.get('exit_reason', 'N/A')}

SKILLS
------
Primary Skills: {', '.join(resume_dict.get('skills', [])) if resume_dict.get('skills') else 'N/A'}
Additional Skills: {', '.join(resume_dict.get('may_also_known_skills', [])) if resume_dict.get('may_also_known_skills') else 'N/A'}
Labels: {', '.join(resume_dict.get('labels', [])) if resume_dict.get('labels') else 'N/A'}

PROFESSIONAL EXPERIENCE
----------------------
{chr(10).join([f'''
Company: {exp.get('company', 'N/A')}
Title: {exp.get('title', 'N/A')}
Duration: {exp.get('from_date', 'N/A')} to {exp.get('until', 'Present') if exp.get('until') else 'Present'}
''' for exp in resume_dict.get('experience', [])]) if resume_dict.get('experience') else 'N/A'}

EDUCATION
---------
{chr(10).join([f'''
Degree: {edu.get('education', 'N/A')}
College: {edu.get('college', 'N/A')}
Pass Year: {edu.get('pass_year', 'N/A')}
''' for edu in resume_dict.get('academic_details', [])]) if resume_dict.get('academic_details') else 'N/A'}

ADDITIONAL INFORMATION
---------------------
Tier 1 MBA: {'Yes' if resume_dict.get('is_tier1_mba') else 'No'}
Tier 1 Engineering: {'Yes' if resume_dict.get('is_tier1_engineering') else 'No'}
Comments: {resume_dict.get('comment', 'N/A')}

PROFESSIONAL LINKS
-----------------
Naukri Profile: {contact_details.get('naukri_profile', 'N/A')}
LinkedIn Profile: {contact_details.get('linkedin_profile', 'N/A')}
Portfolio: {contact_details.get('portfolio_link', 'N/A')}
"""  # Add the combined resume to the dictionary
        resume_dict["combined_resume"] = combined_resume

        # Extract experience titles for skills_titles collection (already normalized and cleaned)
        experience_titles = []
        if "experience" in resume_dict and resume_dict["experience"]:
            for experience in resume_dict["experience"]:
                if "title" in experience and experience["title"]:
                    experience_titles.append(experience["title"])

        # Extract skills for skills_titles collection (already normalized and cleaned)
        skills = []
        if "skills" in resume_dict and resume_dict["skills"]:
            skills.extend(resume_dict["skills"])
        if (
            "may_also_known_skills" in resume_dict
            and resume_dict["may_also_known_skills"]
        ):
            skills.extend(resume_dict["may_also_known_skills"])
            # Generate vector embeddings using our custom vectorizer
        try:
            logging.info("Starting vector embeddings generation")
            resume_with_vectors = add_user_vectorizer.generate_resume_embeddings(
                resume_dict
            )
            logging.info("Vector embeddings generated successfully")
        except Exception as vector_error:
            logging.error(f"Vector generation error: {str(vector_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating vector embeddings: {str(vector_error)}",
            )  # Store in database
        try:
            logging.info("Inserting resume data into database")
            result = collection.insert_one(resume_with_vectors)
            logging.info(f"Successfully inserted resume with ID: {result.inserted_id}")
        except Exception as db_insert_error:
            logging.error(f"Database insertion error: {str(db_insert_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving to database: {str(db_insert_error)}",
            )

        # Add skills and titles to skills_titles collection
        # Ensure skills and experience_titles are normalized before adding to skills_titles collection
        normalized_skills = normalize_text_list(skills) if skills else []
        normalized_experience_titles = (
            normalize_text_list(experience_titles) if experience_titles else []
        )

        try:
            skills_ops.add_multiple_skills(normalized_skills)
            skills_ops.add_multiple_titles(normalized_experience_titles)
            logging.info(f"Added normalized skills: {normalized_skills}")
            logging.info(
                f"Added normalized experience titles: {normalized_experience_titles}"
            )
        except Exception as skills_error:
            logging.warning(f"Skills/titles insertion error: {str(skills_error)}")
            # Don't fail the whole operation for skills insertion errors

        return {
            "status": "success",
            "message": "Resume details saved successfully with vector embeddings",
            "combined_resume": combined_resume,
            "resume_id": str(result.inserted_id),
        }

    except HTTPException as http_ex:
        # Re-raise HTTPException as-is (don't modify it)
        logging.info(f"HTTPException raised: {http_ex.detail}")
        raise http_ex

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(f"Error in submit_resume_details: {error_details}")

        # Return detailed error information
        detail_message = f"Error saving resume details: {error_details['error_type']}: {error_details['error_message']}"
        if not error_details["error_message"]:
            detail_message = f"Error saving resume details: {error_details['error_type']} - Check logs for details"

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail_message,
        )
