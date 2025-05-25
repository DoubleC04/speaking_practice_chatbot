import os
import pandas as pd
import chardet
import csv
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ..utils.log_config import setup_logging
from ..utils.load_config import load_config

logger = setup_logging()

class VectorStoreManager:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config["rag"]["embedding_model"])
        self.persist_directory = self.config["data"]["vector_store"]
        self.chunk_size = self.config["rag"]["chunk_size"]
        self.chunk_overlap = self.config["rag"]["chunk_overlap"]
        self.vectorstore = Chroma(
                    collection_name=self.config["rag"]["chroma_collection"],
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
        
    def initialize_vectorstore(self, pdf_paths=None, csv_paths=None):
        """Initialize vectorstore from PDF and CSV files.

        Args:
            pdf_paths (List, optional): List of PDF file path. Defaults to None.
            csv_paths (List, optional): List of CSV file path. Defaults to None.

        Returns:
            Chroma: Vectorstore is initialized
        """
        logger.info("Initialize Vectorstore...")
        
        self.vectorstore = Chroma(
            collection_name=self.config["rag"]["chroma_collection"],
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        documents = []
        if pdf_paths:
            for pdf_path in pdf_paths:
                documents.extend(self._process_pdf(pdf_path))
        if csv_paths:
            for csv_path in csv_paths:
                documents.extend(self._process_csv(csv_path))
                
        if documents:
            batch_size = 5000
            for i in range(0, len(documents), batch_size):
                batch = documents[i: i + batch_size]
                logger.info(f"Adding batch {i // batch_size + 1}: {len(batch)} documents")
                self.vectorstore.add_documents(batch)
            
        return self.vectorstore
    
    def add_new_documents(self, file_path, file_type=""):
        """
        Add new documents to the vectorstore from a file.

        Args:
            file_path (str): Path to the file to process.
            file_type (str, optional): Type of file ("csv" or "pdf"). Defaults to "".

        Returns:
            Chroma: Updated vectorstore.
        """
        logger.info(f"Add new documents: {file_path} ({file_type})")
        
        if not self.vectorstore:
            logger.error(f"Vectorstore is not initialized.")
            raise ValueError("Vectorstore is not initialized.")
            
        if file_type not in ["csv", "pdf"]:
            logger.error(f"File type is not support: {file_type}")
            raise ValueError(f"File type is not support: {file_type}")
        
        existing_docs = self.vectorstore.get()
        existing_sources = {doc.get('file_name') for doc in existing_docs['metadatas'] if doc.get('file_name')}
        file_name = os.path.basename(file_path)
        
        if file_name in existing_sources:
            logger.info(f"Skipping {file_type} file {file_name}: already processed")
            return self.vectorstore
        
        if file_type == "csv":
            documents = self._process_csv(file_path)
        elif file_type == "pdf":
            documents = self._process_pdf(file_path)
            
        if documents:
            if file_type == "csv":
                existing_words = {
                    doc.get("word")
                    for doc in existing_docs['metadatas']
                    if doc.get("source") == "csv" and doc.get("word")
                }
                documents = [
                    doc for doc in documents
                    if doc.metadata.get("source") != "csv" or doc.metadata.get("word") not in existing_words
                ]
                logger.info(f"After filtering duplicates, {len(documents)} documents remain from {file_path}")
                
            if documents:
                batch_size = 5000
                for i in range(0, len(documents), batch_size):
                    batch = documents[i: i + batch_size]
                    logger.info(f"Adding batch {i // batch_size + 1}: {len(batch)} documents")
                    self.vectorstore.add_documents(batch)
                logger.info(f"Added {len(documents)} new documents from {file_path}")
            else:
                logger.info(f"No new documents to add from {file_path} after filtering")
        else:
            logger.info(f"No documents generated from {file_path}")
        
        return self.vectorstore
    
    def update_documents(self, chunk_id=None, file_name=None, new_content=None, new_metadata=None):
        logger.info(f"Updating documents: chunk_id={chunk_id}, file_name={file_name}")
        
        if not self.vectorstore:
            logger.error("Vectorstore is not initialized.")
            raise ValueError("Vectorstore is not initialized.")
        
        updated_document = Document(
            page_content=new_content,
            metadata = new_metadata,
        )
        
        self.vectorstore.update_documents(document_id=chunk_id, document=updated_document)
        
        return self.vectorstore
    
    def delete_documents(self, chunk_id=None, file_name=None):
        logger.info(f"Deleting documents: chunk_id={chunk_id}, file_name={file_name}")
        
        if not self.vectorstore:
            logger.error("Vectorstore is not initialized.")
            raise ValueError("Vectorstore is not initialized.")
        
        self.vectorstore.delete(ids=chunk_id)
        
        return self.vectorstore
        
    
    def _process_pdf(self, pdf_path):
        """
        Process a PDF file into smaller chunks of documents.

        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            List[Document]: List of text chunks with metadata.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
        except Exception as e:
            logger.exception(f"Failed to load PDF: {e}")
            return []
        
        for doc in data:
            doc.page_content = " ".join(doc.page_content.split())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function = len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_documents(data)
        
        file_name = os.path.basename(pdf_path)
        for i, doc in enumerate(tqdm(documents, desc=f"Adding metadata for {file_name}")):
            doc.metadata.update({
                "chunk_id": f"pdf_{file_name}_{i}",
                "topic": "unknown",
                "level": "unknown",
                "language": "unknown"
            })
            
            unwanted_keys = ['encryptfiltername', 'islinearized', 'iscollectionpresent', 'issignaturespresent', 'custom', 'title', 'author', 'producer', 'creator']
            for key in unwanted_keys:
                if key in doc.metadata:
                    del doc.metadata[key]
            
        logger.info(f"Successfully processed {len(documents)} documents from {file_name}")
        return documents
    
    def _process_csv(self, csv_path):
        """
        Process a CSV file into smaller chunks of documents.

        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            List[Document]: List of text chunks with metadata.
        """
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return []
        
        file_name = os.path.basename(csv_path)
        logger.info(f"Processing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        documents = [
            Document(
                page_content=f"Word: {row.word}\nDefinition: {row.definition}\nPronunciation: {row.pronunciation}",
                metadata={
                    "source": csv_path,
                    "chunk_id": f"csv_{file_name}_{row.Index}",
                    "topic": "unknown",
                    "level": "unknown",
                    "language": "unknown"
                }
            )
            for row in df.itertuples(index=True)
        ]
                    
        logger.info(f"Successfully processed {len(documents)} documents from {file_name}")
                    
        return documents
    
    
    def get_vectorstore(self):
        if not self.vectorstore:
            logger.info(f"Vectorstore not in memory. Attempting to load from {self.persist_directory}")
            try:
                self.vectorstore = Chroma(
                    collection_name=self.config["rag"]["chroma_collection"],
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
                
                existing_docs = self.vectorstore.get()
                if not existing_docs.get("ids", []):
                    logger.error(f"No data found in vectorstore at {self.persist_directory}")
                    raise ValueError("Vectorstore is not initialized or contains no data.")
                
                logger.info(f"Successfully loaded vectorstore with {len(existing_docs['ids'])} documents")
            except Exception as e:
                logger.error(f"Failed to load vectorstore from {self.persist_directory}: {str(e)}")
                raise ValueError(f"Failed to load vectorstore: {str(e)}")
        return self.vectorstore