from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM
from core.custom_logger import CustomLogger
import time

# Initialize logger
logger = CustomLogger().get_logger("ollama_test")

# Initialize router
router = APIRouter()


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask Ollama")
    model: str = Field(default="qwen:4b", description="Ollama model to use")
    temperature: float = Field(
        default=0.7, description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens in response"
    )


class OllamaResponse(BaseModel):
    question: str
    answer: str
    model: str
    response_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None


# Global Ollama LLM instance
ollama_llm = None


def get_ollama_llm(model: str = "qwen:4b", temperature: float = 0.7) -> OllamaLLM:
    """Get or create Ollama LLM instance"""
    global ollama_llm

    try:
        # Create new instance for each request to ensure proper model/temperature
        ollama_llm = OllamaLLM(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434",  # Default Ollama URL
            timeout=60,  # Increase timeout to 60 seconds
            request_timeout=60,
        )
        logger.info(
            f"Ollama LLM initialized with model: {model}, temperature: {temperature}"
        )
        return ollama_llm
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize Ollama: {str(e)}"
        )


@router.post("/ollama/ask", response_model=OllamaResponse)
async def ask_ollama(request: QuestionRequest) -> OllamaResponse:
    """
    Ask a question to Ollama and get a response

    This endpoint allows users to interact with Ollama by asking questions
    and receiving AI-generated responses.
    """
    start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        logger.info(f"Received question: {request.question}")
        logger.info(f"Using model: {request.model}, temperature: {request.temperature}")

        # Get Ollama LLM instance
        llm = get_ollama_llm(model=request.model, temperature=request.temperature)

        # Generate response
        logger.info("Generating response from Ollama...")
        answer = llm.invoke(request.question)

        response_time = time.time() - start_time
        logger.info(f"Response generated in {response_time:.2f} seconds")

        return OllamaResponse(
            question=request.question,
            answer=answer,
            model=request.model,
            response_time=response_time,
            timestamp=timestamp,
            success=True,
        )

    except Exception as e:
        response_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Error processing question: {error_msg}")

        return OllamaResponse(
            question=request.question,
            answer="",
            model=request.model,
            response_time=response_time,
            timestamp=timestamp,
            success=False,
            error_message=error_msg,
        )


@router.get("/ollama/models")
async def get_available_models() -> Dict[str, Any]:
    """
    Get list of available Ollama models
    """
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            return {"success": True, "models": models, "total_models": len(models)}
        else:
            return {
                "success": False,
                "error": "Failed to fetch models from Ollama",
                "models": [],
            }
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return {"success": False, "error": str(e), "models": []}


@router.get("/ollama/health")
async def check_ollama_health() -> Dict[str, Any]:
    """
    Check if Ollama service is running and healthy
    """
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {
                "success": True,
                "status": "healthy",
                "message": "Ollama service is running",
            }
        else:
            return {
                "success": False,
                "status": "unhealthy",
                "message": f"Ollama returned status code: {response.status_code}",
            }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "message": f"Cannot connect to Ollama: {str(e)}",
        }


@router.post("/ollama/chat")
async def chat_with_ollama(request: QuestionRequest) -> Dict[str, Any]:
    """
    Simple chat interface with Ollama
    Returns a more conversational response format
    """
    start_time = time.time()

    try:
        logger.info(f"Chat request: {request.question}")

        # Get Ollama LLM instance
        llm = get_ollama_llm(model=request.model, temperature=request.temperature)

        # Create a chat-friendly prompt
        chat_prompt = f"""You are a helpful AI assistant. Please provide a friendly and informative response to the following question:

Question: {request.question}

Response:"""

        # Generate response
        answer = llm.invoke(chat_prompt)
        response_time = time.time() - start_time

        return {
            "success": True,
            "response": answer,
            "model": request.model,
            "response_time_seconds": round(response_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Chat error: {e}")

        return {
            "success": False,
            "response": "Sorry, I encountered an error while processing your request.",
            "error": str(e),
            "model": request.model,
            "response_time_seconds": round(response_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


# Simple test endpoints
@router.get("/ollama/test")
async def test_ollama() -> Dict[str, Any]:
    """
    Simple test endpoint to verify Ollama is working
    """
    test_question = "Hello! Please introduce yourself in one sentence."

    try:
        llm = get_ollama_llm()
        answer = llm.invoke(test_question)

        return {
            "success": True,
            "test_question": test_question,
            "test_answer": answer,
            "message": "Ollama is working correctly!",
        }
    except Exception as e:
        return {
            "success": False,
            "test_question": test_question,
            "test_answer": "",
            "error": str(e),
            "message": "Ollama test failed",
        }


@router.get("/ollama/test-ui")
async def ollama_test_ui():
    """
    Redirect to the Ollama testing UI
    """
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/ollama_test.html")


@router.get("/ollama/diagnose")
async def diagnose_ollama() -> Dict[str, Any]:
    """
    Comprehensive Ollama diagnostic endpoint
    """
    import requests
    from datetime import datetime

    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "recommendations": [],
    }

    # Test 1: Basic connectivity
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            diagnosis["tests"]["connectivity"] = {
                "status": "success",
                "message": "Ollama is accessible",
                "available_models": models,
            }

            # Check if qwen:4b is available
            if "qwen:4b" in models:
                diagnosis["tests"]["model_availability"] = {
                    "status": "success",
                    "message": "qwen:4b model is available",
                }
            else:
                diagnosis["tests"]["model_availability"] = {
                    "status": "error",
                    "message": "qwen:4b model not found",
                    "available_models": models,
                }
                diagnosis["recommendations"].append(
                    "Install qwen:4b model: ollama pull qwen:4b"
                )
        else:
            diagnosis["tests"]["connectivity"] = {
                "status": "error",
                "message": f"Ollama returned status {response.status_code}",
            }
            diagnosis["recommendations"].append("Check Ollama service status")
    except Exception as e:
        diagnosis["tests"]["connectivity"] = {
            "status": "error",
            "message": f"Cannot connect to Ollama: {str(e)}",
        }
        diagnosis["recommendations"].append("Start Ollama service: ollama serve")

    # Test 2: LLM functionality (only if connectivity works)
    if diagnosis["tests"].get("connectivity", {}).get("status") == "success":
        try:
            llm = get_ollama_llm()
            test_response = llm.invoke("Say 'test successful'")
            diagnosis["tests"]["llm_functionality"] = {
                "status": "success",
                "message": "LLM responds correctly",
                "test_response": str(test_response)[:100],  # First 100 chars
            }
        except Exception as e:
            diagnosis["tests"]["llm_functionality"] = {
                "status": "error",
                "message": f"LLM test failed: {str(e)}",
            }
            diagnosis["recommendations"].append(
                "Check Ollama model loading and system resources"
            )

    # Test 3: RAG system (only if LLM works)
    if diagnosis["tests"].get("llm_functionality", {}).get("status") == "success":
        try:
            from Rag.runner import initialize_rag_app

            rag_app = initialize_rag_app()
            diagnosis["tests"]["rag_initialization"] = {
                "status": "success",
                "message": "RAG system initialized successfully",
            }
        except Exception as e:
            diagnosis["tests"]["rag_initialization"] = {
                "status": "error",
                "message": f"RAG initialization failed: {str(e)}",
            }
            diagnosis["recommendations"].append(
                "Check MongoDB connection and embeddings model"
            )

    # Overall status
    failed_tests = [
        test
        for test, result in diagnosis["tests"].items()
        if result.get("status") == "error"
    ]

    if not failed_tests:
        diagnosis["overall_status"] = "healthy"
        diagnosis["message"] = "All systems operational"
    else:
        diagnosis["overall_status"] = "unhealthy"
        diagnosis["message"] = f"Issues found in: {', '.join(failed_tests)}"

    return diagnosis
