from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
import psutil
import os
from mangodatabase.client import get_collection, get_skills_titles_collection
from mangodatabase.search_indexes import SearchIndexManager

from core.custom_logger import CustomLogger

# Initialize logger
logger = CustomLogger().get_logger("healthcheck")


def verify_vector_search_index(collection, index_name="vector_search_index"):
    """Verify if vector search index exists and is ready"""
    try:
        search_manager = SearchIndexManager()
        return search_manager.check_search_index_exists(index_name)
    except Exception as e:
        logger.error(f"Error verifying vector search index: {str(e)}")
        return False


router = APIRouter(
    prefix="/health",
    tags=["Health Check"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Resume API",
        "version": "1.0.0",
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with all components"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Resume API",
        "version": "1.0.0",
        "components": {},
    }

    overall_status = True

    # Check FastAPI application
    try:
        health_status["components"]["fastapi"] = {
            "status": "healthy",
            "details": {"python_version": sys.version, "platform": sys.platform},
        }
    except Exception as e:
        health_status["components"]["fastapi"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = False

    # Check MongoDB main collection
    try:
        collection = get_collection()
        # Test connection with a simple operation
        collection.database.client.admin.command("ping")

        # Get collection stats
        stats = collection.database.command("collstats", collection.name)

        health_status["components"]["mongodb_main"] = {
            "status": "healthy",
            "details": {
                "database": collection.database.name,
                "collection": collection.name,
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "indexes": stats.get("nindexes", 0),
            },
        }
    except Exception as e:
        health_status["components"]["mongodb_main"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = False

    # Check MongoDB skills_titles collection
    try:
        skills_titles_collection = get_skills_titles_collection()
        # Test connection
        skills_titles_collection.database.client.admin.command("ping")

        # Get collection stats
        stats = skills_titles_collection.database.command(
            "collstats", skills_titles_collection.name
        )

        health_status["components"]["mongodb_skills_titles"] = {
            "status": "healthy",
            "details": {
                "database": skills_titles_collection.database.name,
                "collection": skills_titles_collection.name,
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "indexes": stats.get("nindexes", 0),
            },
        }
    except Exception as e:
        health_status["components"]["mongodb_skills_titles"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = False

    # Check Vector Search Index
    try:
        collection = get_collection()
        vector_index_status = verify_vector_search_index(collection)

        health_status["components"]["vector_search_index"] = {
            "status": "healthy" if vector_index_status else "unhealthy",
            "details": {
                "index_exists": vector_index_status,
                "index_name": "vector_search_index",
            },
        }

        if not vector_index_status:
            overall_status = False

    except Exception as e:
        health_status["components"]["vector_search_index"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = False

    # Check system resources
    try:
        health_status["components"]["system"] = {
            "status": "healthy",
            "details": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (
                    psutil.disk_usage("/").percent
                    if os.name != "nt"
                    else psutil.disk_usage("C:\\").percent
                ),
                "uptime_seconds": int(
                    datetime.utcnow().timestamp() - psutil.boot_time()
                ),
            },
        }
    except Exception as e:
        health_status["components"]["system"] = {"status": "unhealthy", "error": str(e)}
        overall_status = False

    # Set overall status
    health_status["status"] = "healthy" if overall_status else "unhealthy"

    # Return appropriate HTTP status code
    if overall_status:
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(
            content=health_status, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/mongodb")
async def mongodb_health_check():
    """MongoDB specific health check"""
    try:
        # Check main collection
        collection = get_collection()
        collection.database.client.admin.command("ping")

        # Check skills_titles collection
        skills_titles_collection = get_skills_titles_collection()
        skills_titles_collection.database.client.admin.command("ping")

        # Get database info
        db_info = collection.database.client.admin.command("hello")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "main_collection": {
                    "database": collection.database.name,
                    "collection": collection.name,
                    "connection": "active",
                },
                "skills_titles_collection": {
                    "database": skills_titles_collection.database.name,
                    "collection": skills_titles_collection.name,
                    "connection": "active",
                },
                "server_info": {
                    "mongodb_version": db_info.get("version", "unknown"),
                    "is_master": db_info.get("ismaster", False),
                    "max_wire_version": db_info.get("maxWireVersion", 0),
                },
            },
        }
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB health check failed: {str(e)}",
        )


@router.get("/vector-search")
async def vector_search_health_check():
    """Vector Search specific health check"""
    try:
        collection = get_collection()

        # Check if vector search index exists and is ready
        vector_index_status = verify_vector_search_index(collection)

        if vector_index_status:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "index_name": "vector_search_index",
                    "index_status": "ready",
                    "collection": collection.name,
                    "database": collection.database.name,
                },
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector search index is not ready or does not exist",
            )

    except Exception as e:
        logger.error(f"Vector search health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search health check failed: {str(e)}",
        )


@router.get("/system")
async def system_health_check():
    """System resources health check"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/") if os.name != "nt" else psutil.disk_usage("C:\\")

        # Define thresholds
        cpu_threshold = 90
        memory_threshold = 90
        disk_threshold = 90

        # Check if any resource is above threshold
        status = "healthy"
        warnings = []

        if cpu_percent > cpu_threshold:
            status = "warning"
            warnings.append(f"High CPU usage: {cpu_percent}%")

        if memory.percent > memory_threshold:
            status = "warning"
            warnings.append(f"High memory usage: {memory.percent}%")

        if disk.percent > disk_threshold:
            status = "warning"
            warnings.append(f"High disk usage: {disk.percent}%")

        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "cpu": {"percent": cpu_percent, "threshold": cpu_threshold},
                "memory": {
                    "percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "threshold": memory_threshold,
                },
                "disk": {
                    "percent": disk.percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "threshold": disk_threshold,
                },
            },
            "warnings": warnings,
        }

    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"System health check failed: {str(e)}",
        )


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if all critical components are ready
        collection = get_collection()
        collection.database.client.admin.command("ping")

        skills_titles_collection = get_skills_titles_collection()
        skills_titles_collection.database.client.admin.command("ping")

        # Check vector search index
        vector_index_status = verify_vector_search_index(collection)

        if vector_index_status:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector search index not ready",
            )

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}",
        )


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(datetime.now().timestamp() - psutil.boot_time()),
    }
