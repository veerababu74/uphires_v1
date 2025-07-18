import os
import json
import re
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, EmailStr, HttpUrl

# Import core configuration and logging
from core.config import config
from core.custom_logger import CustomLogger
from core.exceptions import LLMProviderError
from core.llm_config import LLMConfigManager, LLMProvider
from core.llm_factory import LLMFactory

# Disable LangSmith tracing to prevent 403 errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# Load environment variables
load_dotenv()

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("multiple_resume_parser")


# Configuration from environment variables
def get_groq_config():
    """Get Groq configuration from environment"""
    return {
        "primary_model": os.getenv("GROQ_PRIMARY_MODEL", "gemma2-9b-it"),
        "backup_model": os.getenv("GROQ_BACKUP_MODEL", "llama-3.1-70b-versatile"),
        "fallback_model": os.getenv("GROQ_FALLBACK_MODEL", "mixtral-8x7b-32768"),
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "1024")),
        "request_timeout": int(os.getenv("GROQ_REQUEST_TIMEOUT", "60")),
    }


def get_ollama_config():
    """Get Ollama configuration from environment"""
    return {
        "primary_model": os.getenv("OLLAMA_PRIMARY_MODEL", "llama3.2:3b"),
        "backup_model": os.getenv("OLLAMA_BACKUP_MODEL", "qwen2.5:3b"),
        "fallback_model": os.getenv("OLLAMA_FALLBACK_MODEL", "qwen:4b"),
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "1024")),
        "api_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
        "timeout": int(os.getenv("OLLAMA_RESPONSE_TIMEOUT", "30")),
    }


def get_api_keys() -> List[str]:
    """Get Groq API keys from environment variables."""
    api_keys_str = os.getenv("GROQ_API_KEYS", "")
    if not api_keys_str:
        # Try legacy environment variable
        api_keys_str = config.GROQ_API_KEY

    if not api_keys_str:
        logger.warning("No Groq API keys found in environment variables")
        return []

    api_keys = api_keys_str.split(",")
    clean_keys = [key.strip() for key in api_keys if key.strip()]
    logger.info(f"Found {len(clean_keys)} Groq API keys")
    return clean_keys


# ===== Pydantic Models =====
class Experience(BaseModel):
    company: str  # Required
    title: str  # Required
    from_date: str  # Required, format: 'YYYY-MM'
    to: Optional[str] = None  # Optional, format: 'YYYY-MM'


class Education(BaseModel):
    education: str  # Required
    college: str  # Required
    pass_year: int  # Required


class ContactDetails(BaseModel):
    name: str  # Required
    email: EmailStr  # Required
    phone: str  # Required
    alternative_phone: Optional[str] = None
    current_city: str  # Required
    looking_for_jobs_in: List[str]  # Required
    gender: Optional[str] = None
    age: Optional[int] = None
    naukri_profile: Optional[str] = None
    linkedin_profile: Optional[str] = None
    portfolio_link: Optional[str] = None
    pan_card: str  # Required
    aadhar_card: Optional[str] = None  # Optional


class Resume(BaseModel):
    user_id: str
    username: str
    contact_details: ContactDetails
    total_experience: Optional[str] = None  # ✅ Already changed to string

    notice_period: Optional[str] = None  # e.g., "Immediate", "30 days"
    currency: Optional[str] = None  # e.g., "INR", "USD"
    pay_duration: Optional[str] = None  # e.g., "monthly", "yearly"
    current_salary: Optional[float] = None
    hike: Optional[float] = None
    expected_salary: Optional[float] = None  # Changed from required to optional
    skills: List[str]
    may_also_known_skills: List[str]
    labels: Optional[List[str]] = None  # Added = None for consistency
    experience: Optional[List[Experience]] = None
    academic_details: Optional[List[Education]] = None
    source: Optional[str] = None  # Source of the resume (e.g., "LinkedIn", "Naukri")
    last_working_day: Optional[str] = None  # Should be ISO format date string
    is_tier1_mba: Optional[bool] = None
    is_tier1_engineering: Optional[bool] = None
    comment: Optional[str] = None
    exit_reason: Optional[str] = None


# ===== Resume Parser Class =====
class ResumeParser:
    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """Initialize ResumeParser with configurable LLM provider.

        Args:
            llm_provider (str, optional): LLM provider ('groq', 'ollama', 'openai', 'google', 'huggingface').
                                        If None, uses LLM_PROVIDER from config.
            api_keys (List[str], optional): List of API keys for API-based providers.
        """
        # Initialize LLM configuration manager
        self.llm_manager = LLMConfigManager()

        # Determine which LLM provider to use
        if llm_provider is None:
            # Use config default
            self.provider = self.llm_manager.provider
        else:
            provider_map = {
                "ollama": LLMProvider.OLLAMA,
                "groq": LLMProvider.GROQ_CLOUD,
                "groq_cloud": LLMProvider.GROQ_CLOUD,
                "openai": LLMProvider.OPENAI,
                "google": LLMProvider.GOOGLE_GEMINI,
                "gemini": LLMProvider.GOOGLE_GEMINI,
                "google_gemini": LLMProvider.GOOGLE_GEMINI,
                "huggingface": LLMProvider.HUGGINGFACE,
                "hf": LLMProvider.HUGGINGFACE,
            }

            if llm_provider.lower() not in provider_map:
                raise LLMProviderError(f"Unsupported LLM provider: {llm_provider}")

            self.provider = provider_map[llm_provider.lower()]
            self.llm_manager.provider = self.provider

        logger.info(f"Initializing Multiple Resume Parser with {self.provider.value}")

        # Initialize using the centralized LLM factory
        try:
            self.llm = LLMFactory.create_llm(force_provider=self.provider)
            logger.info(f"Successfully initialized LLM with {self.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise LLMProviderError(f"Failed to initialize LLM: {e}")

        # Initialize other components
        self.api_keys = api_keys or []
        self.api_usage = {}
        self.current_key_index = 0

        # Setup processing chain
        self.processing_chain = self._setup_processing_chain()

    def _setup_processing_chain(self):
        """Set up the LangChain processing chain for the current provider."""
        try:
            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """Extract resume information strictly as JSON:
                {format_instructions}

                If experience is present, always calculate total experience.
                Ensure the following fields are included: name, email, address, LinkedIn. If any are missing, add random placeholder values.

                RESUME INPUT:
                {resume_text}

                Return ONLY valid JSON without any additional text or explanations.
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(f"Processing chain setup complete for {self.provider.value}")
            return prompt | self.llm | parser

        except Exception as e:
            logger.error(f"Failed to setup processing chain: {str(e)}")
            raise LLMProviderError(f"Failed to setup processing chain: {str(e)}")

    def switch_provider(self, new_provider: str, api_keys: List[str] = None):
        """Switch between LLM providers dynamically.

        Args:
            new_provider (str): New provider to use ('groq', 'ollama', 'openai', 'google', 'huggingface')
            api_keys (List[str], optional): API keys for API-based providers
        """
        logger.info(
            f"Switching LLM provider from {self.provider.value} to {new_provider}"
        )

        # Reinitialize with new provider
        self.__init__(llm_provider=new_provider, api_keys=api_keys)

    def _setup_ollama_chain(self):
        """Set up the LangChain processing chain for Ollama."""
        try:
            # Use Ollama local model with optimized settings for speed and reliability
            model = OllamaLLM(
                model=self.ollama_config["primary_model"],
                temperature=self.ollama_config["temperature"],
                format="json",
                # Optimize for speed and consistency
                num_predict=self.ollama_config["num_predict"],  # Limit response length
                top_k=20,  # Reduce sampling space
                top_p=0.8,  # Focus on most likely tokens
                repeat_penalty=1.1,  # Prevent repetition
                timeout=self.ollama_config["timeout"],  # timeout from config
            )
        except Exception as e:
            logger.error(
                f"Error initializing primary Ollama model ({self.ollama_config['primary_model']}): {e}"
            )
            try:
                # Try backup model
                model = OllamaLLM(
                    model=self.ollama_config["backup_model"],
                    temperature=self.ollama_config["temperature"],
                    format="json",
                    num_predict=self.ollama_config["num_predict"],
                    top_k=20,
                    top_p=0.8,
                    repeat_penalty=1.1,
                    timeout=self.ollama_config["timeout"],
                )
                logger.info(f"Using backup model: {self.ollama_config['backup_model']}")
            except Exception as e2:
                logger.error(f"Backup model also failed: {e2}")
                # Final fallback without advanced parameters
                model = OllamaLLM(
                    model=self.ollama_config["primary_model"],
                    temperature=self.ollama_config["temperature"],
                )

        # Simplified and more direct prompt for better JSON consistency
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

        # Create chain
        chain = prompt | model | parser
        return chain

    def _setup_groq_chain(self, api_key: str):
        """Set up the LangChain processing chain for Groq."""
        if not api_key:
            raise ValueError("API key cannot be empty for Groq.")

        model = ChatGroq(
            temperature=self.groq_config["temperature"],
            model_name=self.groq_config["primary_model"],
            api_key=api_key,
            max_tokens=self.groq_config["max_tokens"],
            request_timeout=self.groq_config["request_timeout"],
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

    def _clean_and_parse_json(self, response) -> dict:
        """Simplified JSON cleaning and parsing."""
        try:
            # Handle different response types
            if isinstance(response, dict):
                return response

            if not isinstance(response, str):
                response = str(response)

            # Remove common prefixes and suffixes
            response = response.strip()

            # Remove markdown code blocks
            if "```json" in response:
                response = re.sub(r"```json\s*", "", response)
                response = re.sub(r"```\s*$", "", response)
            elif "```" in response:
                response = re.sub(r"```[^`]*", "", response)
                response = re.sub(r"```", "", response)

            # Find JSON object boundaries
            start = response.find("{")
            if start == -1:
                raise ValueError("No JSON object found")

            # Find the matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(response[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            if brace_count != 0:
                # Try to balance braces
                response = response[start:] + "}" * brace_count
                end = len(response)

            json_str = response[start:end]

            # Basic cleaning
            json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)  # Remove trailing commas
            json_str = re.sub(
                r"[\x00-\x1f\x7f-\x9f]", "", json_str
            )  # Remove control chars

            # Parse JSON
            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing failed: {e}")
            # Return a basic structure instead of complex fallback
            return {
                "error": "JSON parsing failed",
                "raw_response": str(response)[:200],
                "fallback_used": True,
            }

    def _parse_resume(self, resume_text: str) -> dict:
        """Parse resume text and return structured data."""
        try:
            # Truncate very long resumes to improve speed
            if len(resume_text) > 5000:
                resume_text = resume_text[:5000] + "..."
                logger.info("Resume text truncated for faster processing")

            logger.info("Invoking Ollama model...")
            start_time = time.time()

            raw_output = self.processing_chain.invoke({"resume_text": resume_text})

            end_time = time.time()
            logger.info(f"Ollama response time: {end_time - start_time:.2f} seconds")
            logger.debug(f"Raw output (first 500 chars): {str(raw_output)[:500]}")

            # Use simplified JSON parsing
            result = self._clean_and_parse_json(raw_output)

            # Check if parsing failed
            if "error" in result:
                logger.warning("JSON parsing failed, using fallback parser")
                return self._create_fallback_resume(resume_text)

            # Post-process and validate result
            return self._post_process_result(result)

        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            return {
                "error": "Exception during parsing",
                "exception": str(e),
                "fallback_result": self._create_fallback_resume(resume_text),
            }

    def process_resume(self, resume_text: str) -> Dict:
        """Process a resume using the configured LLM provider."""
        if not resume_text or not resume_text.strip():
            error_msg = "Resume text cannot be empty"
            logger.error(error_msg)
            return {"error": error_msg}

        try:
            # Use the centralized processing logic
            logger.debug(f"Processing resume with {self.provider.value}")
            parsed_data = self._parse_resume(resume_text)

            if "error" not in parsed_data:
                logger.info(
                    f"Successfully processed resume using {self.provider.value}"
                )
                return parsed_data
            else:
                logger.error(
                    f"{self.provider.value} processing failed: {parsed_data.get('error')}"
                )
                return parsed_data

        except Exception as e:
            error_msg = f"{self.provider.value} processing error: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "suggestion": f"Check {self.provider.value} configuration and availability",
                "fallback_result": self._create_fallback_resume(resume_text),
            }

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
        if self.use_ollama:
            return False  # No rotation needed for Ollama

        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            self.processing_chain = self._setup_groq_chain(
                self.api_keys[self.current_key_index]
            )
            logger.info(
                f"Switched to new API key: {self.api_keys[self.current_key_index]}"
            )
            return True
        return False

    def _create_fallback_resume(self, resume_text: str) -> dict:
        """Create a basic resume structure when JSON parsing fails."""
        # Simple regex patterns to extract basic information
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = (
            r"(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\b[0-9]{10}\b)"
        )

        # Extract basic info
        emails = re.findall(email_pattern, resume_text)
        phones = re.findall(phone_pattern, resume_text)

        # Extract name (assume first line or first capitalized words)
        lines = resume_text.strip().split("\n")
        name = "Unknown Candidate"
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if (
                line
                and len(line.split()) <= 4
                and any(word[0].isupper() for word in line.split() if word)
            ):
                name = line
                break

        return {
            "name": name,
            "contact_details": {
                "email": emails[0] if emails else "placeholder@example.com",
                "phone": phones[0] if phones else "+1 123-456-7890",
                "address": "Address not found",
                "linkedin": "https://www.linkedin.com/in/placeholder",
            },
            "education": [],
            "experience": [],
            "projects": [],
            "total_experience": "Experience calculation failed",
            "skills": [],
            "parsing_note": "Fallback parsing used due to JSON parsing failure",
        }

    def _post_process_result(self, result: dict) -> dict:
        """Clean up the parsed result to fix common issues."""
        try:
            if not isinstance(result, dict):
                return result

            # Fix contact details if they contain too much text
            if "contact_details" in result and isinstance(
                result["contact_details"], dict
            ):
                contact = result["contact_details"]

                # Clean phone number - extract just the number
                if "phone" in contact and isinstance(contact["phone"], str):
                    phone_text = contact["phone"]
                    # Extract just phone numbers
                    phone_matches = re.findall(
                        r"(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\b[0-9]{10}\b)",
                        phone_text,
                    )
                    if phone_matches:
                        contact["phone"] = phone_matches[0]
                    elif len(phone_text) > 50:  # If too long, truncate
                        contact["phone"] = "Phone number extraction failed"

                # Clean address - extract just address portion
                if "address" in contact and isinstance(contact["address"], str):
                    address_text = contact["address"]
                    if (
                        len(address_text) > 200
                    ):  # If too long, extract just the beginning
                        # Try to find address-like content
                        address_match = re.search(
                            r"([^,]*,\s*[^,]*,\s*[^,]*)", address_text
                        )
                        if address_match:
                            contact["address"] = address_match.group(1).strip()
                        else:
                            contact["address"] = address_text[:100] + "..."

            # Ensure required fields exist with defaults
            if "name" not in result or not result["name"]:
                result["name"] = "Name not found"

            if "total_experience" not in result:
                result["total_experience"] = "Experience calculation needed"

            if "skills" not in result:
                result["skills"] = []

            if "education" not in result:
                result["education"] = []

            if "experience" not in result:
                result["experience"] = []

            if "projects" not in result:
                result["projects"] = []

            return result

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return result

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False

    def _validate_ollama_model(self, model_name: str) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return any(model_name in model for model in available_models)
            return False
        except Exception:
            return False

    def _get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            import requests

            response = requests.get(
                self.ollama_config["api_url"] + "/api/tags", timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting available Ollama models: {e}")
            return []

    def switch_provider(self, new_provider: str, api_keys: List[str] = None):
        """Switch between LLM providers dynamically.

        Args:
            new_provider (str): New provider to use ('groq', 'ollama')
            api_keys (List[str], optional): API keys for Groq provider
        """
        logger.info(
            f"Switching LLM provider from {'Ollama' if self.use_ollama else 'Groq'} to {new_provider}"
        )

        # Reinitialize with new provider
        self.__init__(use_ollama=(new_provider.lower() == "ollama"), api_keys=api_keys)


def main():
    """Main function to demonstrate resume parsing."""
    sample_resume = """
  RESUME YADAV PANAKJ INDRESHKUMAR Email: yadavanush1234@gmail.com Phone: 9023891599 C -499, umiyanagar behind taxshila school Vastral road – ahmedabad -382418 CareerObjective Todevelop career with an organization which provides me excellent opportunity and enable me tolearn skill to achive organization's goal Personal Details  Full Name : YADAV PANKAJ INDRESHKUMAR  Date of Birth : 14/05/1993  Gender : male  Marital Status : Married  Nationality : Indian  Languages Known : Hindi, English, Gujarati  Hobbies : Reading Work Experience  I Have Two Years Experience (BHARAT PETROLEUM ) As Oil Department Supervisor  I Have ONE Years Experience ( H D B FINACE SERVICES ) As Sales Executive  I Have One Years Experience (MAY GATE SOFTWARE ) As Sales Executive  I Have One Years Experience ( BY U Me – SHOREA SOFECH PRIVATE LTD ) As Sales Executive Education Details Pass Out 2008 - CGPA/Percentage : 51.00% G.S.E.B Pass Out 2010 - CGPA/Percentage : 55.00% G.H.S.E.B Pass Out 2022 – Running Gujarat.uni Interests/Hobbies Listening, music, traveling Declaration I hereby declare that all the details furnished above are true to the best of my knowledge andbelief. Date://2019Place: odhav
    """

    try:
        # Option to run performance test
        import sys

        if len(sys.argv) > 1 and sys.argv[1] == "test":
            test_ollama_speed()
            return

        # Use Ollama by default
        print("Initializing Resume Parser with optimized Ollama settings...")
        parser = ResumeParser(use_ollama=True)

        print("Processing resume...")
        start_time = time.time()
        result = parser.process_resume(sample_resume)
        end_time = time.time()

        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        print("\n" + "=" * 50)
        print("RESUME PARSING RESULT:")
        print("=" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Check if parsing was successful
        if "error" not in result:
            print("\n✅ Resume parsed successfully!")
        elif "parsing_note" in result or "fallback_used" in result:
            print("\n⚠️ Used fallback parsing due to JSON issues")
        else:
            print(f"\n❌ Parsing failed: {result.get('error', 'Unknown error')}")

        # Suggest running performance test
        print(f"\n💡 Tip: Run 'python {sys.argv[0]} test' to test performance")

    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Ollama is installed and running")
        print("2. Start Ollama: ollama serve")
        print("3. Pull required model: ollama pull llama3.2:3b")

    except Exception as e:
        print(f"Error initializing or running parser: {str(e)}")
        print("\nTrying fallback method...")
        try:
            # If Ollama fails, try with a simple fallback
            parser = ResumeParser(use_ollama=True)
            fallback_result = parser._create_fallback_resume(sample_resume)
            print(json.dumps(fallback_result, indent=2, ensure_ascii=False))
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")


def test_ollama_speed():
    """Test function to measure Ollama response speed."""
    test_resume = """
    John Doe
    Email: john.doe@email.com
    Phone: +1-555-123-4567
    
    EXPERIENCE:
    Software Engineer at TechCorp (2020-2023)
    - Developed web applications using Python and React
    - Led team of 3 developers
    
    EDUCATION:
    Bachelor of Computer Science, MIT (2016-2020)
    
    SKILLS: Python, JavaScript, React, SQL
    """

    try:
        print("Testing Ollama speed and reliability...")
        parser = ResumeParser(use_ollama=True)

        # Run multiple tests
        total_time = 0
        successful_parses = 0

        for i in range(3):
            print(f"\nTest {i+1}/3:")
            start_time = time.time()
            result = parser.process_resume(test_resume)
            end_time = time.time()

            response_time = end_time - start_time
            total_time += response_time

            if "error" not in result or "fallback_used" not in result:
                successful_parses += 1
                print(f"✅ Success in {response_time:.2f}s")
                print(f"Extracted name: {result.get('name', 'N/A')}")
                print(
                    f"Extracted email: {result.get('contact_details', {}).get('email', 'N/A')}"
                )
            else:
                print(
                    f"❌ Failed in {response_time:.2f}s: {result.get('error', 'Unknown error')}"
                )

        avg_time = total_time / 3
        success_rate = (successful_parses / 3) * 100

        print(f"\n{'='*50}")
        print(f"PERFORMANCE SUMMARY:")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"{'='*50}")

        if avg_time < 10 and success_rate > 80:
            print("✅ Performance looks good!")
        elif avg_time > 15:
            print("⚠️ Response time is slow. Consider using a smaller model.")
        elif success_rate < 80:
            print("⚠️ Success rate is low. Check model compatibility.")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
    test_ollama_speed()
