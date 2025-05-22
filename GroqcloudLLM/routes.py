from fastapi import APIRouter, HTTPException
import os
from fastapi import File, UploadFile, HTTPException, Request
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from .text_extraction import extract_and_clean_text, clean_text
from .main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from database.operations import ResumeOperations, SkillsTitlesOperations
from database.client import get_collection, get_skills_titles_collection
from core.vectorizer import Vectorizer

# Initialize your parser with API keys (replace with your actual keys)


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
logging.basicConfig(
    filename="cleanup.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


@router.post("/grouqcloud/")
async def extract_clean_text_llam3_3b(file: UploadFile = File(...)):
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
            resume_parser["total_experience"] = 0

        # Calculate experience
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        # Fix the experience titles extraction
        experience_titles = []
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience:
                    experience_titles.append(experience["title"])

        # Fix the skills extraction
        skills = []
        if "skills" in resume_parser:
            skills = resume_parser["skills"]

        resume_ops.create_resume(resume_parser)
        skills_ops.add_multiple_skills(skills)
        skills_ops.add_multiple_titles(experience_titles)

        logging.info(f"Added skills: {skills}")
        logging.info(f"Added experience titles: {experience_titles}")

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
            resume_parser["total_experience"] = 0

        # Calculate experience
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        # Extract experience titles
        experience_titles = []
        if "experience" in resume_parser:
            for experience in resume_parser["experience"]:
                if "title" in experience:
                    experience_titles.append(experience["title"])

        # Extract skills
        skills = []
        if "skills" in resume_parser:
            skills = resume_parser["skills"]

        # Store in database
        resume_ops.create_resume(resume_parser)
        skills_ops.add_multiple_skills(skills)
        skills_ops.add_multiple_titles(experience_titles)

        logging.info(f"Added skills: {skills}")
        logging.info(f"Added experience titles: {experience_titles}")

        return {
            "cleaned_text": cleaned_text,
            "resume_parser": resume_parser,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
