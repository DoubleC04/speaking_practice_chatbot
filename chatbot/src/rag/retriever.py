from src.utils.log_config import setup_logging
from src.utils.load_config import load_config
from src.rag.vector_store import VectorStoreManager

logger = setup_logging()

class Retriever:
    def __init__(self):
        """
        Initialize the Retriever for RAG system with supports for multiple retrievers.
        """
        self.config = load_config("config.yaml")
        self.vector_store_manager = VectorStoreManager()
        self.retrievers = {}
        
    def initialize_retriever(self, query_type="simple"):
        """
        Initialize a retriever for the specified query type.

        Args:
            query_type (str): Type of query ("simple" or "complex"). Defaults to "simple".

        Returns:
            Retriever: The initialized retriever object
        """
        logger.info("Initialize Retriever...")
        
        vectorstore = self.vector_store_manager.get_vectorstore()
        k = self.config["rag"]["retriever_k"][query_type]
        retriever = vectorstore.as_retriever(
            search_kwargs = {"k": k}
        )
        self.retrievers[query_type] = retriever
        return retriever
    
    def get_retriever(self, query_type="simple"):
        """
        Get the retriever for the specified query type, initializing it if not already done.

        Args:
            query_type (str): Type of query ("simple" or "complex"). Defaults to "simple".

        Returns:
            Retriever: The retriever object
        """
        if query_type not in self.retrievers:
            self.initialize_retriever(query_type)
        return self.retrievers[query_type]
        