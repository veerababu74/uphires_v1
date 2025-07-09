"""
LLM Configuration API
====================

API endpoints for managing LLM configuration and provider switching.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from core.llm_config import get_llm_config, LLMProvider, configure_llm_provider
from core.llm_factory import LLMFactory, get_current_provider, switch_provider
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("llm_config_api")

router = APIRouter(
    prefix="/llm-config",
    tags=["LLM Configuration"],
    responses={404: {"description": "Not found"}},
)


class ProviderSwitchRequest(BaseModel):
    """Request to switch LLM provider"""

    provider: str = Field(..., description="Provider name: 'ollama' or 'groq_cloud'")


class LLMTestRequest(BaseModel):
    """Request to test LLM functionality"""

    provider: Optional[str] = Field(None, description="Optional provider to test")
    prompt: Optional[str] = Field("Hello, respond with 'OK'", description="Test prompt")


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration"""

    ollama_model: Optional[str] = Field(None, description="Ollama model to use")
    groq_model: Optional[str] = Field(None, description="Groq model to use")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature setting"
    )


@router.get("/status", summary="Get LLM Configuration Status")
async def get_llm_status() -> Dict[str, Any]:
    """
    Get current LLM configuration status including:
    - Current provider
    - Provider availability
    - Configuration details
    """
    try:
        config_manager = get_llm_config()
        status = LLMFactory.get_provider_status()

        # Add configuration details
        status["configuration"] = config_manager.get_status()
        status["validated"] = config_manager._validated

        return {
            "success": True,
            "data": status,
            "message": "LLM status retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Failed to get LLM status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM status: {str(e)}",
        )


@router.post("/switch-provider", summary="Switch LLM Provider")
async def switch_llm_provider(request: ProviderSwitchRequest) -> Dict[str, Any]:
    """
    Switch between LLM providers (Ollama or Groq Cloud)
    """
    try:
        old_provider = get_current_provider()
        success = switch_provider(request.provider)

        if success:
            new_provider = get_current_provider()
            logger.info(f"Successfully switched from {old_provider} to {new_provider}")

            return {
                "success": True,
                "data": {
                    "old_provider": old_provider,
                    "new_provider": new_provider,
                    "switch_time": "immediate",
                },
                "message": f"Successfully switched to {new_provider}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to switch to provider: {request.provider}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error switching provider: {str(e)}",
        )


@router.post("/test", summary="Test LLM Connection")
async def test_llm(request: LLMTestRequest) -> Dict[str, Any]:
    """
    Test LLM connection and functionality
    """
    try:
        provider = None
        if request.provider:
            if request.provider.lower() == "ollama":
                provider = LLMProvider.OLLAMA
            elif request.provider.lower() in ["groq", "groq_cloud"]:
                provider = LLMProvider.GROQ_CLOUD
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider: {request.provider}",
                )

        # Test connection
        connection_success = LLMFactory.test_llm_connection(provider)

        if connection_success:
            # Try a simple generation
            try:
                llm = LLMFactory.create_llm(force_provider=provider)
                response = (
                    llm.invoke(request.prompt)
                    if hasattr(llm, "invoke")
                    else llm(request.prompt)
                )

                return {
                    "success": True,
                    "data": {
                        "connection": "successful",
                        "provider": (
                            provider.value if provider else get_current_provider()
                        ),
                        "test_prompt": request.prompt,
                        "response": (
                            str(response)[:200] + "..."
                            if len(str(response)) > 200
                            else str(response)
                        ),
                    },
                    "message": "LLM test completed successfully",
                }

            except Exception as e:
                return {
                    "success": False,
                    "data": {
                        "connection": "failed",
                        "provider": (
                            provider.value if provider else get_current_provider()
                        ),
                        "error": str(e),
                    },
                    "message": "LLM connection test failed",
                }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM connection test failed",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing LLM: {str(e)}",
        )


@router.get("/providers", summary="List Available Providers")
async def list_providers() -> Dict[str, Any]:
    """
    List all available LLM providers and their capabilities
    """
    try:
        status_info = LLMFactory.get_provider_status()

        providers = [
            {
                "name": "ollama",
                "display_name": "Ollama (Local)",
                "description": "Local LLM server running on your machine",
                "available": status_info["ollama"]["available"],
                "models": status_info["ollama"]["models"],
                "error": status_info["ollama"]["error"],
                "benefits": [
                    "Free to use",
                    "Privacy-focused (local)",
                    "No API limits",
                    "Works offline",
                ],
                "requirements": [
                    "Ollama installed locally",
                    "Compatible models downloaded",
                    "Sufficient system resources",
                ],
            },
            {
                "name": "groq_cloud",
                "display_name": "Groq Cloud",
                "description": "High-speed cloud-based LLM API",
                "available": status_info["groq"]["available"],
                "api_keys_count": status_info["groq"]["api_keys_count"],
                "error": status_info["groq"]["error"],
                "benefits": [
                    "Very fast inference",
                    "No local resources needed",
                    "Multiple model options",
                    "Highly reliable",
                ],
                "requirements": [
                    "Valid API key(s)",
                    "Internet connection",
                    "API rate limits apply",
                ],
            },
        ]

        return {
            "success": True,
            "data": {
                "current_provider": status_info["current_provider"],
                "providers": providers,
            },
            "message": "Providers listed successfully",
        }

    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing providers: {str(e)}",
        )


@router.get("/config", summary="Get Current Configuration")
async def get_current_config() -> Dict[str, Any]:
    """
    Get detailed current configuration for the active provider
    """
    try:
        config_manager = get_llm_config()
        active_config = config_manager.get_active_config()

        if config_manager.is_ollama_enabled():
            config_data = {
                "provider": "ollama",
                "api_url": active_config.api_url,
                "primary_model": active_config.primary_model,
                "backup_model": active_config.backup_model,
                "temperature": active_config.temperature,
                "max_context_length": active_config.max_context_length,
                "timeout": active_config.response_timeout,
                "advanced_settings": {
                    "num_predict": active_config.num_predict,
                    "top_k": active_config.top_k,
                    "top_p": active_config.top_p,
                    "repeat_penalty": active_config.repeat_penalty,
                },
            }
        else:
            config_data = {
                "provider": "groq_cloud",
                "primary_model": active_config.primary_model,
                "backup_model": active_config.backup_model,
                "temperature": active_config.temperature,
                "max_tokens": active_config.max_tokens,
                "max_context_length": active_config.max_context_length,
                "api_keys_count": len(active_config.api_keys),
                "current_key_index": active_config.current_key_index,
                "rate_limits": {
                    "requests_per_minute": active_config.requests_per_minute,
                    "tokens_per_minute": active_config.tokens_per_minute,
                },
            }

        return {
            "success": True,
            "data": config_data,
            "message": "Configuration retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting configuration: {str(e)}",
        )


@router.post("/validate", summary="Validate Configuration")
async def validate_configuration() -> Dict[str, Any]:
    """
    Validate current LLM configuration
    """
    try:
        config_manager = get_llm_config()
        is_valid = config_manager.validate_configuration()

        validation_details = {
            "overall_valid": is_valid,
            "provider": config_manager.provider.value,
            "checks": {},
        }

        if config_manager.is_ollama_enabled():
            validation_details["checks"] = {
                "ollama_connection": False,
                "models_available": False,
                "primary_model_exists": False,
            }

            try:
                # Check Ollama connection
                import requests

                response = requests.get(
                    f"{config_manager.ollama_config.api_url}/api/tags", timeout=5
                )
                validation_details["checks"]["ollama_connection"] = (
                    response.status_code == 200
                )

                if response.status_code == 200:
                    models_data = response.json()
                    models = [model["name"] for model in models_data.get("models", [])]
                    validation_details["checks"]["models_available"] = len(models) > 0
                    validation_details["checks"]["available_models"] = models

                    # Check if primary model exists
                    primary_model = config_manager.ollama_config.primary_model
                    validation_details["checks"]["primary_model_exists"] = any(
                        primary_model in model for model in models
                    )

            except Exception as e:
                validation_details["checks"]["connection_error"] = str(e)

        else:
            validation_details["checks"] = {
                "api_keys_configured": len(config_manager.groq_config.api_keys) > 0,
                "api_key_format_valid": False,
            }

            if config_manager.groq_config.api_keys:
                current_key = config_manager.groq_config.get_current_api_key()
                validation_details["checks"]["api_key_format_valid"] = (
                    current_key and len(current_key.strip()) > 10
                )

        return {
            "success": True,
            "data": validation_details,
            "message": "Configuration validation completed",
        }

    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating configuration: {str(e)}",
        )


@router.get("/health", summary="LLM Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Quick health check for LLM services
    """
    try:
        config_manager = get_llm_config()
        current_provider = config_manager.provider.value

        # Test current provider
        test_result = LLMFactory.test_llm_connection()

        health_status = {
            "healthy": test_result,
            "provider": current_provider,
            "timestamp": "now",  # You might want to use actual timestamp
            "details": {},
        }

        if config_manager.is_ollama_enabled():
            try:
                import requests

                response = requests.get(
                    f"{config_manager.ollama_config.api_url}/api/version", timeout=2
                )
                if response.status_code == 200:
                    health_status["details"]["ollama_version"] = response.json()
            except:
                health_status["details"]["ollama_version"] = "unavailable"

        return {
            "success": True,
            "data": health_status,
            "message": "Health check completed",
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "data": {"healthy": False, "error": str(e)},
            "message": "Health check failed",
        }
