from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from .models import BestMatchResult, AllMatchesResult
from .config import RAGConfig


class ChainManager:
    """Manages LangChain chains for RAG operations"""

    def __init__(self, llm: ChatGroq):
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
Based on the user's question and the following candidate data snippets, identify the single best matching candidate.
Return ONLY the MongoDB '_id' of the best matching candidate in JSON format.

{format_instructions}

Candidate Data Snippets:
---
{context}
---

User Question: {question}

Best Matching Candidate:"""

        output_parser = JsonOutputParser(pydantic_object=BestMatchResult)
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

        ranking_output_parser = JsonOutputParser(pydantic_object=AllMatchesResult)
        ranking_prompt = PromptTemplate(
            template=ranking_prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": ranking_output_parser.get_format_instructions()
            },
        )
        self.ranking_chain = ranking_prompt | self.llm | ranking_output_parser
