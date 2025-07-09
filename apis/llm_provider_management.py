"""
Global LLM Provider Management API
=================================

This module provides API endpoints for managing LLM providers globally
across the entire application.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from core.llm_provider_switch import provider_switch_manager
from core.custom_logger import CustomLogger

# Initialize router and logger
router = APIRouter(prefix="/llm-provider", tags=["LLM Provider Management"])
logger = CustomLogger().get_logger("llm_provider_api")


@router.post("/switch")
async def switch_llm_provider_global(
    provider: str = Query(..., description="LLM provider to use ('groq' or 'ollama')"),
    api_keys: Optional[List[str]] = Query(
        None, description="Groq API keys (only required for Groq provider)"
    ),
) -> Dict[str, Any]:
    """
    Switch LLM provider globally across the entire application.

    This endpoint allows you to switch between Groq Cloud and Ollama providers
    for all resume parsing and text processing operations.

    Args:
        provider: The LLM provider to use ('groq' or 'ollama')
        api_keys: List of Groq API keys (only required when switching to Groq)

    Returns:
        Dict containing switch status and provider information
    """
    try:
        logger.info(f"Received request to switch to {provider} provider")

        if provider.lower() not in ["groq", "ollama"]:
            raise HTTPException(
                status_code=400, detail="Invalid provider. Must be 'groq' or 'ollama'"
            )

        # Validate requirements for Groq
        if provider.lower() == "groq" and not api_keys:
            # Check if API keys are already configured
            current_status = provider_switch_manager.get_provider_status()
            if provider.lower() != current_status.get(
                "current_provider"
            ) and not current_status.get("has_valid_keys", False):
                raise HTTPException(
                    status_code=400,
                    detail="API keys are required when switching to Groq provider",
                )

        # Perform the switch
        result = provider_switch_manager.switch_provider_globally(
            provider.lower(), api_keys
        )

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])

        logger.info(f"Successfully switched to {provider} provider")
        return result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to switch LLM provider: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/status")
async def get_llm_provider_status() -> Dict[str, Any]:
    """
    Get the current status of the LLM provider configuration.

    Returns comprehensive information about the currently active LLM provider,
    including configuration details, service status, and available models.

    Returns:
        Dict containing provider status and configuration information
    """
    try:
        status = provider_switch_manager.get_provider_status()
        logger.info(
            f"Provider status requested - current: {status.get('current_provider')}"
        )
        return status

    except Exception as e:
        error_msg = f"Failed to get LLM provider status: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/test-connection")
async def test_llm_provider_connection(
    provider: Optional[str] = Query(
        None,
        description="Provider to test ('groq' or 'ollama'). If not specified, tests current provider.",
    )
) -> Dict[str, Any]:
    """
    Test connection to the specified LLM provider.

    This endpoint allows you to verify that the LLM provider is accessible
    and properly configured before switching to it.

    Args:
        provider: Optional provider to test. If not specified, tests current provider.

    Returns:
        Dict containing connection test results
    """
    try:
        if provider and provider.lower() not in ["groq", "ollama"]:
            raise HTTPException(
                status_code=400, detail="Invalid provider. Must be 'groq' or 'ollama'"
            )

        result = provider_switch_manager.test_provider_connection(provider)

        if result["status"] == "error":
            # Don't raise HTTP exception for connection errors - return the error info
            logger.warning(
                f"Connection test failed for {result.get('provider')}: {result.get('message')}"
            )
        else:
            logger.info(f"Connection test successful for {result.get('provider')}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to test LLM provider connection: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/providers")
async def list_available_providers() -> Dict[str, Any]:
    """
    List all available LLM providers and their configurations.

    Returns information about all supported LLM providers, including
    their current configuration status and requirements.

    Returns:
        Dict containing information about available providers
    """
    try:
        current_status = provider_switch_manager.get_provider_status()
        current_provider = current_status.get("current_provider", "unknown")

        providers_info = {
            "current_provider": current_provider,
            "available_providers": {
                "ollama": {
                    "name": "Ollama",
                    "type": "local",
                    "description": "Local LLM service running on your machine",
                    "requires_api_keys": False,
                    "requires_service": True,
                    "default_models": ["llama3.2:3b", "qwen2.5:3b", "qwen:4b"],
                    "is_current": current_provider == "ollama",
                },
                "groq": {
                    "name": "Groq Cloud",
                    "type": "api",
                    "description": "Cloud-based LLM service with high-performance inference",
                    "requires_api_keys": True,
                    "requires_service": False,
                    "default_models": [
                        "gemma2-9b-it",
                        "llama-3.1-70b-versatile",
                        "mixtral-8x7b-32768",
                    ],
                    "is_current": current_provider == "groq",
                },
            },
        }

        # Add current provider details
        if current_provider == "ollama" and "service_status" in current_status:
            providers_info["available_providers"]["ollama"]["service_status"] = (
                current_status["service_status"]
            )
            providers_info["available_providers"]["ollama"]["available_models"] = (
                current_status.get("available_models", [])
            )
        elif current_provider == "groq" and "has_valid_keys" in current_status:
            providers_info["available_providers"]["groq"]["has_valid_keys"] = (
                current_status["has_valid_keys"]
            )
            providers_info["available_providers"]["groq"]["api_keys_count"] = (
                current_status.get("api_keys_count", 0)
            )

        return providers_info

    except Exception as e:
        error_msg = f"Failed to list available providers: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/config")
async def get_llm_provider_config() -> Dict[str, Any]:
    """
    Get detailed configuration for the current LLM provider.

    Returns comprehensive configuration details for the currently active
    LLM provider, including model settings and performance parameters.

    Returns:
        Dict containing detailed configuration information
    """
    try:
        status = provider_switch_manager.get_provider_status()
        current_provider = status.get("current_provider")

        if current_provider == "ollama":
            config_info = {
                "provider": "ollama",
                "type": "local",
                "model": status.get("model"),
                "backup_model": status.get("backup_model"),
                "api_url": status.get("api_url"),
                "connection_timeout": status.get("connection_timeout"),
                "request_timeout": status.get("request_timeout"),
                "service_status": status.get("service_status"),
                "available_models": status.get("available_models", []),
            }
        else:
            config_info = {
                "provider": "groq",
                "type": "api",
                "model": status.get("model"),
                "backup_model": status.get("backup_model"),
                "api_keys_count": status.get("api_keys_count"),
                "has_valid_keys": status.get("has_valid_keys"),
                "request_timeout": status.get("request_timeout"),
                "max_tokens": status.get("max_tokens"),
            }

        config_info["configuration_valid"] = status.get("configuration_valid", False)

        return config_info

    except Exception as e:
        error_msg = f"Failed to get LLM provider configuration: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
