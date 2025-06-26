import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gemma2-9b-it"
TEMPERATURE = 1

OLLAMA_DEFAULT_MODEL = "qwiqwen:4b"


def get_api_keys() -> List[str]:
    """Get API keys from environment variables."""
    api_keys = os.getenv("GROQ_API_KEYS", "").split(",")
    return [key.strip() for key in api_keys if key.strip()]


# ===== Pydantic Models =====
class Project(BaseModel):
    name: str = Field(default="", description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: List[str] = Field(
        default_factory=list, description="Technologies used"
    )
    role: str = Field(default="", description="Role in the project")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="", description="End date")
    duration: str = Field(default="", description="Total duration")


class ContactDetails(BaseModel):
    email: str = Field(
        default="placeholder@example.com", description="Candidate's email address"
    )
    phone: str = Field(default="+1 123-456-7890", description="Contact phone number")
    address: str = Field(
        default="123 Placeholder St, Placeholder City", description="Physical address"
    )
    linkedin: str = Field(
        default="https://www.linkedin.com/in/placeholder",
        description="LinkedIn profile URL",
    )


class Education(BaseModel):
    degree: str = Field(default="", description="Degree or qualification")
    institution: str = Field(default="", description="Educational institution name")
    dates: str = Field(default="", description="Start and end dates")


class Experience(BaseModel):
    title: str = Field(default="", description="Job position title")
    company: str = Field(default="", description="Company/organization name")
    start_date: str = Field(default="", description="Employment start date")
    end_date: str = Field(default="", description="End date or 'Present'")
    duration: str = Field(default="", description="Total duration in role")


class Resume(BaseModel):
    name: str = Field(default="John Doe", description="Full name of candidate")
    contact_details: ContactDetails
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    total_experience: str = Field(
        default="0 years, 0 months", description="Total work experience duration"
    )
    skills: List[str] = Field(
        default_factory=list, description="List of technical/professional skills"
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
        parser = JsonOutputParser(pydantic_object=Resume)

        prompt_template = """Extract resume information strictly as JSON:
            {format_instructions}

            If experience is present, always calculate total experience.
            Ensure the following fields are included: name, email, address, LinkedIn. If any are missing, add random placeholder values.
            If there are projects, include them with details like name, description, technologies, role, start_date, end_date, and duration. Use empty values for missing fields.

            RESUME INPUT:
            {resume_text}

            Return ONLY valid JSON without any additional text or explanations.
            """

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
  RESUME YADAV PANAKJ INDRESHKUMAR Email: yadavanush1234@gmail.com Phone: 9023891599 C -499, umiyanagar behind taxshila school Vastral road – ahmedabad -382418 CareerObjective Todevelop career with an organization which provides me excellent opportunity and enable me tolearn skill to achive organization's goal Personal Details  Full Name : YADAV PANKAJ INDRESHKUMAR  Date of Birth : 14/05/1993  Gender : male  Marital Status : Married  Nationality : Indian  Languages Known : Hindi, English, Gujarati  Hobbies : Reading Work Experience  I Have Two Years Experience (BHARAT PETROLEUM ) As Oil Department Supervisor  I Have ONE Years Experience ( H D B FINACE SERVICES ) As Sales Executive  I Have One Years Experience (MAY GATE SOFTWARE ) As Sales Executive  I Have One Years Experience ( BY U Me – SHOREA SOFECH PRIVATE LTD ) As Sales Executive Education Details Pass Out 2008 - CGPA/Percentage : 51.00% G.S.E.B Pass Out 2010 - CGPA/Percentage : 55.00% G.H.S.E.B Pass Out 2022 – Running Gujarat.uni Interests/Hobbies Listening, music, traveling Declaration I hereby declare that all the details furnished above are true to the best of my knowledge andbelief. Date://2019Place: odhav
    """

    try:
        parser = ResumeParser()
        result = parser.process_resume(sample_resume)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
