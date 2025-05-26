from pathlib import Path
from ..rag.vector_store import VectorStoreManager

vector_store_manager = VectorStoreManager()

pdf_dir = Path("data/pdf")
csv_dir = Path("data/csv")

pdf_paths = [str(file) for file in pdf_dir.glob("*.pdf")]
csv_paths = [str(file) for file in csv_dir.glob("*.csv")]

if not pdf_paths and not csv_paths:
    print("File not found.")
else:
    print("PDF files:", pdf_paths)
    print("CSV files:", csv_paths)
    
    vector_store_manager.initialize_vectorstore(pdf_paths)




