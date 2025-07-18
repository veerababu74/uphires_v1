from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from .models import BestMatchResult, AllMatchesResult
from .config import RAGConfig
import json
import re
from typing import Any, Type, Union
from pydantic import Field
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("chain_manager")


class RobustJsonOutputParser(BaseOutputParser):
    """Custom JSON parser that can handle malformed LLM output"""

    def __init__(self, pydantic_object=None, **kwargs):
        super().__init__(**kwargs)
        # Store the pydantic object using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "_pydantic_object", pydantic_object)

    @property
    def pydantic_object(self):
        """Property to access the pydantic object"""
        return self._pydantic_object

    def parse(self, text: str) -> Any:
        """Parse LLM output with robust JSON cleaning"""
        try:
            # Log the raw input for debugging
            logger.info(f"Raw LLM response type: {type(text)}")
            logger.info(f"Raw LLM response (first 500 chars): {str(text)[:500]}")

            # First try normal JSON parsing if it's already a dict
            if isinstance(text, dict):
                return self.pydantic_object(**text)

            # Handle AIMessage objects from LangChain
            if hasattr(text, "content"):
                logger.info(
                    f"Extracting content from AIMessage: {text.content[:200]}..."
                )
                text = text.content

            # Clean the text from common LLM output issues
            cleaned_text = self._clean_llm_output(str(text))

            # If cleaned text is empty JSON, provide default structure
            if cleaned_text == "{}":
                if self.pydantic_object == AllMatchesResult:
                    logger.warning(
                        "LLM returned empty response, using fallback for AllMatchesResult"
                    )
                    return AllMatchesResult(total_candidates=0, matches=[])
                else:
                    logger.warning(
                        "LLM returned empty response, using fallback for BestMatchResult"
                    )
                    return BestMatchResult(id="no_match_found")

            # Try to parse the cleaned JSON
            parsed_data = json.loads(cleaned_text)

            # Validate the parsed data has required fields
            if self.pydantic_object == AllMatchesResult:
                if not isinstance(parsed_data, dict) or "matches" not in parsed_data:
                    logger.warning("Invalid AllMatchesResult structure, using fallback")
                    return AllMatchesResult(total_candidates=0, matches=[])
            elif self.pydantic_object == BestMatchResult:
                if not isinstance(parsed_data, dict) or (
                    "_id" not in parsed_data and "id" not in parsed_data
                ):
                    logger.warning("Invalid BestMatchResult structure, using fallback")
                    return BestMatchResult(id="no_match_found")

            return self.pydantic_object(**parsed_data)

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(
                f"Raw output: {str(text)[:500]}..."
            )  # Fallback: return appropriate empty structure
            if self.pydantic_object == AllMatchesResult:
                logger.info("Using fallback AllMatchesResult due to parsing error")
                return AllMatchesResult(total_candidates=0, matches=[])
            else:
                logger.info("Using fallback BestMatchResult due to parsing error")
                return BestMatchResult(id="parsing_error")

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output to extract valid JSON - enhanced for extreme malformed cases"""
        try:
            # Handle the case where text is empty or just whitespace
            if not text or not text.strip():
                logger.warning("Empty LLM output received")
                return "{}"

            # Handle single character responses (like "-")
            if len(text.strip()) <= 2:
                logger.warning(f"Text appears to be invalid: {text}...")
                return "{}"

            # Log the raw output for debugging (first 200 chars)
            logger.debug(f"Raw LLM output: {str(text)[:200]}...")

            # Handle extreme cases of repeated markers
            original_text = str(text)

            # Remove all occurrences of ```json and ``` markers aggressively
            # This handles cases where LLM outputs hundreds of repeated markers
            text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
            text = re.sub(r"```\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

            # Remove any leading/trailing non-JSON text
            # Find the first { and last }
            start_idx = text.find("{")
            end_idx = text.rfind("}")

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON braces found in response")
                return "{}"

            text = text[start_idx : end_idx + 1]

            # Remove any trailing explanation text after JSON
            # Split by newlines and take only lines that look like JSON
            lines = text.split("\n")
            json_lines = []
            json_started = False
            brace_count = 0

            for line in lines:
                if "{" in line or json_started:
                    json_started = True
                    json_lines.append(line)
                    brace_count += line.count("{") - line.count("}")
                    if json_started and brace_count <= 0:
                        break

            if json_lines:
                text = "\n".join(json_lines)

            # Clean up common JSON formatting issues
            text = re.sub(r",(\s*[}\]])", r"\1", text)  # Remove trailing commas
            text = re.sub(
                r"[\x00-\x1f\x7f-\x9f]", "", text
            )  # Remove control characters
            text = re.sub(r"\\n", "", text)  # Remove escaped newlines

            # Final validation - try to parse to ensure it's valid JSON
            try:
                parsed = json.loads(text)
                logger.debug("Successfully parsed cleaned JSON")
                return text
            except json.JSONDecodeError as jde:
                logger.warning(f"JSON decode error after cleaning: {jde}")
                logger.warning(f"Problematic text: {text[:200]}...")
                return "{}"

        except Exception as e:
            logger.error(f"Error cleaning LLM output: {e}")
            logger.error(f"Original text sample: {str(text)[:200]}...")
            return "{}"
            text = re.sub(
                r"```json", "", original_text, flags=re.IGNORECASE | re.MULTILINE
            )
            text = re.sub(r"```", "", text, flags=re.IGNORECASE | re.MULTILINE)

            # Remove common LLM output prefixes/suffixes
            text = re.sub(
                r"^(Here's|Here is|The result is|Result:|Response:|JSON Response:).*?[:\n]",
                "",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            # Remove extra whitespace, newlines, and control characters
            text = re.sub(
                r"\s+", " ", text
            )  # Normalize all whitespace to single spaces
            text = text.strip()

            # If text is empty after cleaning, return empty JSON
            if not text:
                logger.warning("Text empty after cleaning markdown markers")
                return "{}"

            # Check if the entire text is just repeated markers or numbers
            if len(text) < 5 or re.match(r"^[\d\s,]+$", text):
                logger.warning(f"Text appears to be invalid: {text[:50]}...")
                return "{}"

            # Try to find JSON object boundaries with greedy matching
            json_match = re.search(
                r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL
            )
            if json_match:
                text = json_match.group(1)
                logger.debug(f"Found JSON object: {text[:100]}...")
            else:
                # If no JSON object found, check for array
                json_array_match = re.search(
                    r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])", text, re.DOTALL
                )
                if json_array_match:
                    text = json_array_match.group(1)
                    logger.debug(f"Found JSON array: {text[:100]}...")
                else:
                    # Last resort: try to extract any quoted strings that might be JSON-like
                    logger.warning(
                        f"No valid JSON structure found in cleaned text: {text[:100]}..."
                    )
                    return "{}"

            # Clean up common JSON formatting issues
            text = re.sub(r",(\s*[}\]])", r"\1", text)  # Remove trailing commas
            text = re.sub(
                r"[\x00-\x1f\x7f-\x9f]", "", text
            )  # Remove control characters
            text = re.sub(r"\\n", "", text)  # Remove escaped newlines

            # Final validation - try to parse to ensure it's valid JSON
            try:
                parsed = json.loads(text)
                logger.debug("Successfully parsed cleaned JSON")
                return text
            except json.JSONDecodeError as jde:
                logger.warning(f"JSON decode error after cleaning: {jde}")
                logger.warning(f"Problematic text: {text[:200]}...")
                return "{}"

        except Exception as e:
            logger.error(f"Error cleaning LLM output: {e}")
            logger.error(f"Original text sample: {str(text)[:200]}...")
            return "{}"

    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM"""
        if self.pydantic_object == AllMatchesResult:
            return """Return a valid JSON object with this exact structure:
{{
  "total_candidates": <number>,
  "matches": [
    {{
      "_id": "<mongodb_id>",
      "relevance_score": <float_between_0_and_1>,
      "match_reason": "<explanation>"
    }}
  ]
}}

IMPORTANT: 
- Return ONLY the JSON object, no additional text or markdown formatting
- Do not wrap in ```json``` code blocks
- Ensure all candidates are included in the matches array
- Scores must be between 0.0 and 1.0"""
        else:
            return """Return a valid JSON object with this exact structure:
{
  "_id": "<mongodb_id>"
}

IMPORTANT: 
- Return ONLY the JSON object, no additional text or markdown formatting
- Do not wrap in ```json``` code blocks"""


class ChainManager:
    """Manages LangChain chains for RAG operations"""

    def __init__(self, llm: BaseLanguageModel):
        """Initialize ChainManager with any LangChain-compatible LLM.

        Args:
            llm: Any LangChain-compatible language model (Groq, Ollama, OpenAI, etc.)
        """
        self.llm = llm
        self.retrieval_chain = None
        self.ranking_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """Setup retrieval and ranking chains"""
        self._setup_retrieval_chain()
        self._setup_ranking_chain()

    def _setup_retrieval_chain(self):
        """Setup the retrieval chain for best match"""
        prompt_template_text = """You are an AI assistant specialized in analyzing candidate resumes to find the best match for job requirements.

TASK: Analyze the provided candidate data and identify the single best matching candidate for the given query.

INSTRUCTIONS:
1. Carefully read the user's query/job requirements
2. Compare each candidate against the requirements
3. Consider: technical skills, experience level, domain expertise, education
4. Return ONLY the MongoDB '_id' of the best matching candidate

VALIDATION RULES:
- If the query is too vague or irrelevant, return: {{"_id": "no_match_found"}}
- If no candidates match the requirements, return: {{"_id": "no_match_found"}}
- Always return valid JSON format

IMPORTANT OUTPUT RULES:
- Response must be ONLY valid JSON
- No explanations, no markdown, no code blocks
- Must be exactly this format: {{"_id": "some_id_here"}}

{format_instructions}

CANDIDATE DATA:
{context}

QUERY: {question}

JSON RESPONSE:"""

        output_parser = RobustJsonOutputParser(pydantic_object=BestMatchResult)
        prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        self.retrieval_chain = prompt | self.llm | output_parser

    def _setup_ranking_chain(self):
        """Setup the ranking chain for all matches"""
        ranking_prompt_template = """You are an AI recruiter ranking candidates for a job position.

TASK: Analyze ALL candidates and rank them by relevance to the job requirements.

CRITICAL: Return ONLY a JSON object with this EXACT structure - no explanations, no extra text:

{{
  "total_candidates": <number_of_candidates>,
  "matches": [
    {{
      "_id": "<candidate_mongodb_id>",
      "relevance_score": <decimal_between_0_and_1>,
      "match_reason": "<brief_explanation>"
    }}
  ]
}}

SCORING CRITERIA (0.0 to 1.0):
- 1.0: Perfect match (all requirements met)
- 0.8-0.9: Excellent match (most requirements met)  
- 0.6-0.7: Good match (some requirements met)
- 0.4-0.5: Partial match (related experience)
- 0.1-0.3: Weak match (minimal relevance)
- 0.0: No match (no relevant experience)

INSTRUCTIONS:
1. Analyze EVERY candidate provided in the context
2. Extract only the "_id" field from each candidate
3. Score each candidate (0.0-1.0) based on job relevance
4. Keep match_reason under 15 words
5. Return ALL candidates in the matches array

CRITICAL OUTPUT RULES:
- Response must be ONLY valid JSON (no markdown, no code blocks)
- Use "_id" field exactly as provided in candidate data
- Scores must be decimal numbers (e.g., 0.8, not "0.8")
- Include ALL candidates in matches array

{format_instructions}

JOB REQUIREMENTS: {question}

CANDIDATES:
{context}

JSON RESPONSE:"""

        ranking_output_parser = RobustJsonOutputParser(pydantic_object=AllMatchesResult)
        ranking_prompt = PromptTemplate(
            template=ranking_prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": ranking_output_parser.get_format_instructions()
            },
        )
        self.ranking_chain = ranking_prompt | self.llm | ranking_output_parser
