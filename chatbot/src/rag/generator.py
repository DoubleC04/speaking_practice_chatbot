from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.log_config import setup_logging
from src.utils.load_config import load_config

logger = setup_logging()

class Generator:
    def __init__(self):
        """
        A class to generate responses using a language model.
        """
        self.config = load_config("config.yaml")
        self.llm = OllamaLLM(
            model=self.config["rag"]["llm_model"],
            base_url="http://127.0.0.1:11434"
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=
            """
            Answer the question based on the following context:
            {context}

            ---
            Answer the question based on the above context:
            {question}
            """ 
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        
    def generate(self, context, question):
        """
        Generate a response for the user's question.

        Args:
            context (str): Conversation context from retriever.
            question (str): User's question

        Returns:
            str: The generated response.
        """
        logger.info(f"Answer the question: {question}")
        response = self.chain.invoke({"context": context, "question": question})
        return response