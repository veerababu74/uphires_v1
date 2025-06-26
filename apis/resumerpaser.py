import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gemma2-9b-it"
TEMPERATURE = 1

OLLAMA_DEFAULT_MODEL = "qwen:4b"


def get_api_keys() -> List[str]:
    """Get API keys from environment variables."""
    api_keys = os.getenv("GROQ_API_KEYS", "").split(",")
    return [key.strip() for key in api_keys if key.strip()]


# ===== Updated Pydantic Models =====
class Experience(BaseModel):
    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    from_date: str = Field(description="Start date in YYYY-MM format")
    to: Optional[str] = Field(
        default=None, description="End date in YYYY-MM format, None if current"
    )


class Education(BaseModel):
    education: str = Field(description="Degree or qualification")
    college: str = Field(description="Educational institution name")
    pass_year: int = Field(description="Year of graduation")


class ContactDetails(BaseModel):
    name: str = Field(description="Full name")
    email: EmailStr = Field(description="Email address")
    phone: str = Field(description="Primary phone number")
    alternative_phone: Optional[str] = Field(
        default=None, description="Alternative phone number"
    )
    current_city: str = Field(description="Current city of residence")
    looking_for_jobs_in: List[str] = Field(
        default_factory=list, description="Cities looking for jobs in"
    )
    gender: Optional[str] = Field(default=None, description="Gender")
    age: Optional[int] = Field(default=None, description="Age")
    naukri_profile: Optional[str] = Field(
        default=None, description="Naukri profile URL"
    )
    linkedin_profile: Optional[str] = Field(
        default=None, description="LinkedIn profile URL"
    )
    portfolio_link: Optional[str] = Field(
        default=None, description="Portfolio website link"
    )
    pan_card: str = Field(description="PAN card number")
    aadhar_card: Optional[str] = Field(default=None, description="Aadhar card number")


class ResumeData(BaseModel):

    contact_details: ContactDetails = Field(description="Contact information")
    total_experience: Optional[str] = Field(
        default=None, description="Total work experience"
    )
    notice_period: Optional[str] = Field(
        default=None, description="Notice period (e.g., Immediate, 30 days)"
    )
    currency: Optional[str] = Field(
        default=None, description="Currency (e.g., INR, USD)"
    )
    pay_duration: Optional[str] = Field(
        default=None, description="Pay duration (e.g., monthly, yearly)"
    )
    current_salary: Optional[float] = Field(default=None, description="Current salary")
    hike: Optional[float] = Field(default=None, description="Expected hike percentage")
    expected_salary: Optional[float] = Field(
        default=None, description="Expected salary"
    )
    skills: List[str] = Field(default_factory=list, description="Primary skills")
    may_also_known_skills: List[str] = Field(
        default_factory=list, description="Additional skills"
    )
    labels: Optional[List[str]] = Field(default=None, description="Labels or tags")
    experience: Optional[List[Experience]] = Field(
        default=None, description="Work experience details"
    )
    academic_details: Optional[List[Education]] = Field(
        default=None, description="Educational background"
    )
    source: Optional[str] = Field(default=None, description="Source of resume")
    last_working_day: Optional[str] = Field(
        default=None, description="Last working day (ISO format)"
    )
    is_tier1_mba: Optional[bool] = Field(
        default=None, description="Whether from tier 1 MBA institute"
    )
    is_tier1_engineering: Optional[bool] = Field(
        default=None, description="Whether from tier 1 engineering institute"
    )
    comment: Optional[str] = Field(default=None, description="Additional comments")
    exit_reason: Optional[str] = Field(
        default=None, description="Reason for leaving current job"
    )


# ===== Resume Parser Class =====
class ResumeParser:
    def __init__(self, api_keys: List[str] = None):
        """Initialize ResumeParser with API keys.

        Args:
            api_keys (List[str], optional): List of API keys. If None, loads from environment.
        """
        if api_keys is None:
            self.api_keys = get_api_keys()
        else:
            self.api_keys = [key.strip() for key in api_keys if key.strip()]

        if not self.api_keys:
            raise ValueError("No API keys provided or found in environment variables.")

        self.api_usage = {key: 0 for key in self.api_keys}
        self.current_key_index = 0
        self.processing_chain = self._setup_processing_chain(
            self.api_keys[self.current_key_index]
        )

    def _setup_processing_chain(self, api_key: str):
        """Set up the LangChain processing chain."""
        if not api_key:
            raise ValueError("API key cannot be empty.")

        # model = ChatGroq(
        #     temperature=TEMPERATURE, model_name=DEFAULT_MODEL, api_key=api_key
        # )
        model = OllamaLLM(
            temperature=TEMPERATURE,
            model=OLLAMA_DEFAULT_MODEL,
        )
        parser = JsonOutputParser(pydantic_object=ResumeData)

        prompt_template = """You are an expert resume parser. Extract information from the resume text and convert it to JSON format.

                FOLLOW THESE RULES STRICTLY:
                1. Return ONLY valid JSON - no explanations, no markdown, no extra text
                2. Use the exact field names provided in the schema
                3. If a field is missing, use the default values specified below

                REQUIRED FIELDS (must be present):
    
                - contact_details.name: Full name from resume
                - contact_details.email: Email address (use "placeholder@example.com" if missing)
                - contact_details.phone: Phone number (use "+91-0000000000" if missing)
                - contact_details.current_city: Extract city from address (use "Unknown" if missing)
                - contact_details.looking_for_jobs_in: Array of cities (use current_city if missing)
                - contact_details.pan_card: PAN number (use "ABCDE1234F" if missing)

                EXPERIENCE EXTRACTION:
                - company: Company name exactly as written
                - title: Job title/position
                - from_date: Convert to "YYYY-MM" format (e.g., "2020-01" for January 2020)
                - to: Convert to "YYYY-MM" format, use null for current jobs

                EDUCATION EXTRACTION:
                - education: Degree name (e.g., "B.Tech", "MBA", "10th Grade")
                - college: Institution name
                - pass_year: Year as integer (e.g., 2020)

                SKILLS EXTRACTION:
                - Extract technical skills, soft skills, programming languages
                - Separate into "skills" (primary) and "may_also_known_skills" (secondary)

                OPTIONAL FIELDS - Set to null if not found:
                - total_experience: Calculate from work history (e.g., "5 years 3 months")
                - current_salary, expected_salary: Extract salary numbers
                - currency: Use "INR" for Indian resumes, "USD" for others
                - notice_period: Extract notice period info
                - age: Calculate from date of birth if available
                - gender: Extract if mentioned

                SCHEMA:
                {format_instructions}

                RESUME TEXT:
                {resume_text}

                JSON OUTPUT:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | model | parser

    def _clean_json_response(self, response) -> str:
        """Clean and extract JSON from response."""
        try:
            if isinstance(response, dict):
                return json.dumps(response)
            elif isinstance(response, str):
                # Remove code block markers and clean whitespace
                cleaned = re.sub(r"```[^`]*```", "", response, flags=re.DOTALL)
                cleaned = cleaned.strip()
                # Extract JSON object if wrapped in text
                match = re.search(r"({[\s\S]*})", cleaned)
                return match.group(1) if match else cleaned
            else:
                return str(response)
        except Exception as e:
            print(f"Error cleaning JSON response: {str(e)}")
            return str(response)

    def _repair_json(self, malformed_json: str) -> str:
        """Repair common JSON formatting issues."""
        try:
            # Remove trailing commas
            repaired = re.sub(r",(\s*[}\]])", r"\1", malformed_json)
            # Remove control characters
            repaired = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", repaired)
            # Balance braces
            open_braces = repaired.count("{")
            close_braces = repaired.count("}")
            if open_braces > close_braces:
                repaired += "}" * (open_braces - close_braces)
            elif close_braces > open_braces:
                repaired = "{" * (close_braces - open_braces) + repaired
            return repaired
        except Exception as e:
            print(f"Error repairing JSON: {str(e)}")
            return malformed_json

    def _extract_json_object(self, text) -> str:
        matches = re.findall(r"({.*?})", text, re.DOTALL)
        if matches:
            return max(matches, key=len)
        return None

    def _repair_and_load_json(self, raw_json) -> dict:
        cleaned_json = self._clean_json_response(raw_json)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            repaired_json = self._repair_json(cleaned_json)
            try:
                return json.loads(repaired_json)
            except Exception:
                extracted_json = self._extract_json_object(cleaned_json)
                if extracted_json:
                    try:
                        return json.loads(extracted_json)
                    except Exception:
                        pass
                return {"error": "Failed to parse JSON", "raw_output": raw_json}

    def _parse_resume(self, resume_text: str) -> dict:
        """Parse resume text and return structured data."""
        try:
            raw_output = self.processing_chain.invoke({"resume_text": resume_text})
            if isinstance(raw_output, dict):
                return raw_output

            cleaned_json = self._clean_json_response(raw_output)
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                repaired_json = self._repair_json(cleaned_json)
                try:
                    return json.loads(repaired_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {str(e)}")
                    return {
                        "error": "Failed to parse response",
                        "raw_output": str(raw_output),
                    }
        except Exception as e:
            print(f"Error parsing resume: {str(e)}")
            return {"error": str(e)}

    def process_resume(self, resume_text: str) -> Dict:
        """Process a resume and handle API key rotation if needed."""
        while True:
            try:
                parsed_data = self._parse_resume(resume_text)
                self.api_usage[self.api_keys[self.current_key_index]] += 1
                return parsed_data
            except Exception as e:
                error_msg = str(e).lower()
                print(
                    f"Error with API key {self.api_keys[self.current_key_index]}: {error_msg}"
                )

                if self._should_rotate_key(error_msg):
                    if not self._rotate_to_next_key():
                        return {
                            "error": "All API keys exhausted",
                            "api_usage": self.api_usage,
                        }
                else:
                    return {"error": "Unexpected error", "details": error_msg}

    def _should_rotate_key(self, error_msg: str) -> bool:
        """Check if we should rotate to the next API key based on error message."""
        rotation_triggers = [
            "rate limit",
            "quota exceeded",
            "too many requests",
            "organization_restricted",
        ]
        return any(trigger in error_msg for trigger in rotation_triggers)

    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available API key."""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            self.processing_chain = self._setup_processing_chain(
                self.api_keys[self.current_key_index]
            )
            print(f"Switched to new API key: {self.api_keys[self.current_key_index]}")
            return True
        return False


def main():
    """Main function to demonstrate resume parsing."""
    sample_resume = """
  RESUME YADAV PANAKJ INDRESHKUMAR Email: yadavanush1234@gmail.com Phone: 9023891599 C -499, umiyanagar behind taxshila school Vastral road – ahmedabad -382418 CareerObjective Todevelop career with an organization which provides me excellent opportunity and enable me tolearn skill to achive organization's goal Personal Details  Full Name : YADAV PANKAJ INDRESHKUMAR  Date of Birth : 14/05/1993  Gender : male  Marital Status : Married  Nationality : Indian  Languages Known : Hindi, English, Gujarati  Hobbies : Reading Work Experience  I Have Two Years Experience (BHARAT PETROLEUM ) As Oil Department Supervisor  I Have ONE Years Experience ( H D B FINACE SERVICES ) As Sales Executive  I Have One Years Experience (MAY GATE SOFTWARE ) As Sales Executive  I Have One Years Experience ( BY U Me – SHOREA SOFECH PRIVATE LTD ) As Sales Executive Education Details Pass Out 2008 - CGPA/Percentage : 51.00% G.S.E.B Pass Out 2010 - CGPA/Percentage : 55.00% G.H.S.E.B Pass Out 2022 – Running Gujarat.uni Interests/Hobbies Listening, music, traveling Declaration I hereby declare that all the details furnished above are true to the best of my knowledge andbelief. Date://2019Place: odhav
    """

    try:
        parser = ResumeParser()
        result = parser.process_resume(sample_resume)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
