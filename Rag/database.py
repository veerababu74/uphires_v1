from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from .config import RAGConfig
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("database_manager")


class DatabaseManager:
    """Manages database connections and vector store initialization"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.client = None
        self.collection = None
        self.vector_store = None
        self._initialize()

    def _initialize(self):
        """Initialize database connection and vector store"""
        self._connect_to_mongodb()
        self._initialize_vector_store()

    def _connect_to_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(RAGConfig.MONGODB_URI)
            db = self.client[RAGConfig.DB_NAME]
            self.collection = db[RAGConfig.COLLECTION_NAME]
            # Test connection
            self.client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database '{RAGConfig.DB_NAME}' and collection '{RAGConfig.COLLECTION_NAME}'"
            )
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    def _initialize_vector_store(self):
        """Initialize MongoDB Atlas Vector Search"""
        if self.embeddings is None or self.collection is None:
            raise ValueError("Embeddings and collection must be initialized first")

        try:
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=RAGConfig.INDEX_NAME,
                text_key="combined_resume",
                embedding_key=RAGConfig.VECTOR_FIELD,
            )
            logger.info(
                f"MongoDB Atlas Vector Search initialized with index '{RAGConfig.INDEX_NAME}' on field '{RAGConfig.VECTOR_FIELD}'"
            )
        except Exception as e:
            logger.error(f"Error initializing MongoDBAtlasVectorSearch: {e}")
            raise
