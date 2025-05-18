from fastapi import APIRouter, HTTPException, status
import os
from fastapi import File, UploadFile, HTTPException, Request
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta, date
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from GroqcloudLLM.main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from database.operations import ResumeOperations, SkillsTitlesOperations
from database.client import get_collection, get_skills_titles_collection
from core.vectorizer import Vectorizer
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sentence_transformers import SentenceTransformer
import numpy as np


class AddUserDataVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        if not text or text == "N/A":
            return np.zeros(384).tolist()
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return np.zeros(384).tolist()

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume"""
        resume_with_vectors = resume_data.copy()

        # Generate experience text and vector
        experience_text = ""
        if "experience" in resume_data:
            experience_texts = []
            for exp in resume_data["experience"]:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('from_date', '')} to {exp.get('until', 'Present')}"
                experience_texts.append(exp_text)
            experience_text = ". ".join(experience_texts)
            resume_with_vectors["experience_text_vector"] = self.generate_embedding(
                experience_text
            )

        # Generate education text and vector
        education_text = ""
        if "academic_details" in resume_data:
            education_texts = []
            for edu in resume_data["academic_details"]:
                edu_text = f"{edu.get('education', '')} from {edu.get('college', '')} ({edu.get('pass_year', '')})"
                education_texts.append(edu_text)
            education_text = ". ".join(education_texts)
            resume_with_vectors["education_text_vector"] = self.generate_embedding(
                education_text
            )

        # Generate skills text and vector (combining primary and additional skills)
        all_skills = resume_data.get("skills", []) + resume_data.get(
            "may_also_known_skills", []
        )
        skills_text = ", ".join(all_skills) if all_skills else ""
        resume_with_vectors["skills_vector"] = self.generate_embedding(skills_text)

        # Generate vector for the combined resume text
        combined_resume = resume_data.get("combined_resume", "")
        resume_with_vectors["combined_resume_vector"] = self.generate_embedding(
            combined_resume
        )

        return resume_with_vectors


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


router = APIRouter(
    prefix="/edit",
    tags=["edit"],
    responses={404: {"description": "Not found"}},
)


class Experience(BaseModel):
    company: str
    title: str
    from_date: date
    until: Optional[date]


class Education(BaseModel):
    education: str
    college: str
    pass_year: int


class ContactDetails(BaseModel):
    name: Optional[str] = None
    email: EmailStr
    phone: str
    alternative_phone: Optional[str] = None
    current_city: str
    looking_for_jobs_in: str
    gender: Optional[str] = None
    age: Optional[int] = None
    naukri_profile: Optional[HttpUrl] = None
    linkedin_profile: Optional[HttpUrl] = None
    portfolio_link: Optional[HttpUrl] = None


class ResumeData(BaseModel):
    user_id: str
    username: str
    contact_details: ContactDetails
    total_experience: float
    notice_period: int
    currency: str
    pay_duration: str
    current_salary: float
    hike: float
    expected_salary: float
    skills: List[str]
    may_also_known_skills: List[str]
    labels: List[str]
    experience: List[Experience]
    academic_details: List[Education]
    source: str
    last_working_day: Optional[date]
    is_tier1_mba: bool
    is_tier1_engineering: bool
    comment: Optional[str]
    exit_reason: Optional[str]


@router.post("/upload")
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
            raise HTTPException(status_code=400, detail="No resume data found.")

        # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0

        # Calculate experience
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        return {
            "filename": file.filename,
            "total_resume_text": total_resume_text,
            "resume_parser": resume_parser,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@router.post("/submit-details")
async def submit_resume_details(resume_data: ResumeData):
    """
    Submit resume details and save to MongoDB with proper vector embeddings
    """
    try:
        # Convert Pydantic model to dictionary
        resume_dict = resume_data.model_dump()
        resume_dict["created_at"] = datetime.utcnow()

        # Convert datetime objects to strings for MongoDB compatibility
        for exp in resume_dict["experience"]:
            exp["from_date"] = exp["from_date"].isoformat()
            if exp["until"]:
                exp["until"] = exp["until"].isoformat()

        # Convert contact details URLs to strings
        contact_details = resume_dict["contact_details"]
        if contact_details.get("naukri_profile"):
            contact_details["naukri_profile"] = str(contact_details["naukri_profile"])
        if contact_details.get("linkedin_profile"):
            contact_details["linkedin_profile"] = str(
                contact_details["linkedin_profile"]
            )
        if contact_details.get("portfolio_link"):
            contact_details["portfolio_link"] = str(contact_details["portfolio_link"])

        # Convert last_working_day to string if present
        if resume_dict.get("last_working_day"):
            resume_dict["last_working_day"] = resume_dict[
                "last_working_day"
            ].isoformat()

        # Combine all data into one proper resume format for text representation
        combined_resume = f"""
RESUME

PERSONAL INFORMATION
-------------------
Name: {contact_details.get('name', 'N/A')}
Contact Details:
  Email: {contact_details['email']}
  Phone: {contact_details['phone']}
  Alternative Phone: {contact_details.get('alternative_phone', 'N/A')}
  Current City: {contact_details['current_city']}
  Looking for jobs in: {contact_details['looking_for_jobs_in']}
  Gender: {contact_details.get('gender', 'N/A')}
  Age: {contact_details.get('age', 'N/A')}

PROFESSIONAL SUMMARY
-------------------
Total Experience: {resume_dict['total_experience']} years
Notice Period: {resume_dict['notice_period']} days
Current Salary: {resume_dict['currency']} {resume_dict['current_salary']} ({resume_dict['pay_duration']})
Expected Salary: {resume_dict['currency']} {resume_dict['expected_salary']} ({resume_dict['pay_duration']})
Hike Expected: {resume_dict['hike']}%
Last Working Day: {resume_dict.get('last_working_day', 'N/A')}
Exit Reason: {resume_dict.get('exit_reason', 'N/A')}

SKILLS
------
Primary Skills: {', '.join(resume_dict['skills'])}
Additional Skills: {', '.join(resume_dict['may_also_known_skills']) if resume_dict['may_also_known_skills'] else 'N/A'}
Labels: {', '.join(resume_dict['labels']) if resume_dict['labels'] else 'N/A'}

PROFESSIONAL EXPERIENCE
----------------------
{chr(10).join([f'''
Company: {exp['company']}
Title: {exp['title']}
Duration: {exp['from_date']} to {exp['until'] if exp['until'] else 'Present'}
''' for exp in resume_dict['experience']])}

EDUCATION
---------
{chr(10).join([f'''
Degree: {edu['education']}
College: {edu['college']}
Pass Year: {edu['pass_year']}
''' for edu in resume_dict['academic_details']])}

ADDITIONAL INFORMATION
---------------------
Tier 1 MBA: {'Yes' if resume_dict['is_tier1_mba'] else 'No'}
Tier 1 Engineering: {'Yes' if resume_dict['is_tier1_engineering'] else 'No'}
Comments: {resume_dict.get('comment', 'N/A')}

PROFESSIONAL LINKS
-----------------
Naukri Profile: {contact_details.get('naukri_profile', 'N/A')}
LinkedIn Profile: {contact_details.get('linkedin_profile', 'N/A')}
Portfolio: {contact_details.get('portfolio_link', 'N/A')}
"""

        # Add the combined resume to the dictionary
        resume_dict["combined_resume"] = combined_resume

        # Extract experience titles for skills_titles collection
        experience_titles = []
        if "experience" in resume_dict:
            for experience in resume_dict["experience"]:
                if "title" in experience:
                    experience_titles.append(experience["title"])

        # Extract skills for skills_titles collection
        skills = []
        if "skills" in resume_dict:
            skills = resume_dict["skills"]
        if "may_also_known_skills" in resume_dict:
            skills.extend(resume_dict["may_also_known_skills"])

        # Generate vector embeddings using our custom vectorizer
        resume_with_vectors = add_user_vectorizer.generate_resume_embeddings(
            resume_dict
        )

        # Store in database
        result = collection.insert_one(resume_with_vectors)

        # Add skills and titles to skills_titles collection
        skills_ops.add_multiple_skills(skills)
        skills_ops.add_multiple_titles(experience_titles)

        logging.info(f"Added skills: {skills}")
        logging.info(f"Added experience titles: {experience_titles}")

        return {
            "status": "success",
            "message": "Resume details saved successfully with vector embeddings",
            "combined_resume": combined_resume,
            "resume_id": str(result.inserted_id),
        }
    except Exception as e:
        logging.error(f"Error in submit_resume_details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving resume details: {str(e)}",
        )
