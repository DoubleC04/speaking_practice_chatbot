from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.utils.log_config import setup_logging

logger = setup_logging()

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever().get_retriever()
        self.generator = Generator()
        self.chain = (
            RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()})
            | (lambda x: {
                "context": "\n".join([doc.page_content for doc in x["context"]]),
                "question": x["question"]
            })
            | self.generator.chain
        )
        
    def rag_invoke(self, question):
        """
        Process the question and generate a response using the RAG pipeline.

        Args:
            question (str): User's question

        Returns:
            str: The generated response
        """
        logger.info(f"Process the question: {question}")
    
        response = self.chain.invoke(question)
        
        return response