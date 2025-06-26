from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from .models import BestMatchResult, AllMatchesResult
from .config import RAGConfig
import json
import re
from typing import Any
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("chain_manager")


class RobustJsonOutputParser(BaseOutputParser):
    """Custom JSON parser that can handle malformed LLM output"""

    def __init__(self, pydantic_object):
        super().__init__()
        self._pydantic_object = pydantic_object

    @property
    def pydantic_object(self):
        """Property to access the pydantic object"""
        return self._pydantic_object

    def parse(self, text: str) -> Any:
        """Parse LLM output with robust JSON cleaning"""
        try:
            # First try normal JSON parsing if it's already a dict
            if isinstance(text, dict):
                return self.pydantic_object(**text)

            # Handle AIMessage objects from LangChain
            if hasattr(text, "content"):
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

            # Log the raw output for debugging (first 200 chars)
            logger.debug(f"Raw LLM output: {str(text)[:200]}...")

            # Handle extreme cases of repeated markers
            original_text = str(text)

            # Remove all occurrences of ```json and ``` markers aggressively
            # This handles cases where LLM outputs hundreds of repeated markers
            text = re.sub(
                r"```json", "", original_text, flags=re.IGNORECASE | re.MULTILINE
            )
            text = re.sub(r"```", "", text, flags=re.IGNORECASE | re.MULTILINE)

            # Remove common LLM output prefixes/suffixes
            text = re.sub(
                r"^(Here's|Here is|The result is|Result:|Response:).*?[:\n]",
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

            # Check if the entire text is just repeated markers (no actual content)
            if (
                len(text) < 5 or text.count("j") > len(text) * 0.3
            ):  # Too many 'j' chars suggests repeated "```json"
                logger.warning("Text appears to be mostly repeated markers")
                return "{}"

            # Try to find JSON object boundaries with greedy matching
            json_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
                logger.debug(f"Found JSON object: {text[:100]}...")
            else:
                # If no JSON object found, check for array
                json_array_match = re.search(r"(\[.*\])", text, re.DOTALL)
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
        if self._pydantic_object == AllMatchesResult:
            return """Return a valid JSON object with this exact structure:
{
  "total_candidates": <number>,
  "matches": [
    {
      "_id": "<mongodb_id>",
      "relevance_score": <float_between_0_and_1>,
      "match_reason": "<explanation>"
    }
  ]
}

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

    def __init__(self, llm: ChatGroq | OllamaLLM):
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
        prompt_template_text = """You are an assistant analyzing candidate resume data to find the best match for a query.
First, analyze if the query is relevant to candidate matching or job description analysis:

1. If the query is about finding candidates based on job description:
   - Validate if the job description is relevant and contains specific requirements
   - If job description is too vague or irrelevant, return "no_match_found"
   - Focus on matching technical skills, experience, and qualifications

2. If the query is about specific candidate information:
   - Validate if the query is relevant to candidate data
   - If query is irrelevant or not related to candidate information, return "no_match_found"
   - Focus on matching specific candidate attributes

3. If the query is irrelevant or not related to either:
   - Return "no_match_found" with explanation

Based on the user's question and the following candidate data snippets, identify the single best matching candidate.
Return ONLY the MongoDB '_id' of the best matching candidate in JSON format.

{format_instructions}

Candidate Data Snippets:
---
{context}
---

User Question: {question}

Best Matching Candidate:"""

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
        ranking_prompt_template = """You are an expert AI recruiter analyzing candidate resumes to rank ALL candidates based on their relevance to a specific search query.

First, validate the query type and relevance:

1. For Job Description Based Queries:
   - Check if the job description is specific and relevant
   - If job description is vague or irrelevant, return empty matches
   - Focus on matching technical requirements, experience, and qualifications

2. For Candidate Information Queries:
   - Verify if the query is related to candidate data
   - If query is irrelevant to candidate information, return empty matches
   - Focus on matching specific candidate attributes

3. For Irrelevant Queries:
   - Return empty matches with explanation

CRITICAL INSTRUCTIONS:
1. You MUST analyze EVERY single candidate document provided in the context (separated by ---DOCUMENT_SEPARATOR---)
2. Score each candidate from 0.0 to 1.0 based on how well they match the query:
   - 1.0 = Perfect match (has all required skills/experience exactly)
   - 0.8-0.9 = Excellent match (has most required skills/experience with good alignment)
   - 0.6-0.7 = Good match (has some required skills/experience with decent alignment)
   - 0.4-0.5 = Partial match (has related skills/experience with some relevance)
   - 0.1-0.3 = Weak match (minimal relevance, few matching criteria)
   - 0.0 = No match (no relevant skills or experience)
3. Provide a clear, specific reason for each score explaining WHY this candidate matches or doesn't match
4. Return ALL candidates ranked by relevance score (highest first)
5. Do not skip any candidates - analyze every single one provided

Consider these factors when scoring (in order of importance):
- Technical skills match (programming languages, frameworks, tools, technologies)
- Experience level and years of experience matching requirements
- Industry/domain experience relevance
- Educational background relevance (degrees, certifications)
- Specific project/work experience alignment
- Notice period and availability
- Salary expectations alignment
- Geographic location/work preferences

{format_instructions}

Search Query: {question}

ALL Candidate Documents to Analyze:
---
{context}
---

Now analyze and rank EVERY SINGLE candidate above:"""

        ranking_output_parser = RobustJsonOutputParser(pydantic_object=AllMatchesResult)
        ranking_prompt = PromptTemplate(
            template=ranking_prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": ranking_output_parser.get_format_instructions()
            },
        )
        self.ranking_chain = ranking_prompt | self.llm | ranking_output_parser
