# app/orchestration.py

from typing import List
from app.document_loader import load_documents_txt
from app.embedder import Embedder
from app.vector_store import VectorStore
from app.generator import AnswerGenerator

def orchestration(query: str) -> str:
    """
    Run the full RAG pipeline for a given query string.
    
    Steps:
    1. Load documents from local storage.
    2. Embed the query and documents.
    3. Build or load the vector store.
    4. Search for similar documents using query embedding.
    5. Generate an answer based on retrieved documents and the query.
    
    Args:
        query (str): The user's input question.
        
    Returns:
        str: The generated answer text.
    """
    # Step 1: Load documents
    docs: List[str] = load_documents_txt("data/sample_docs")
    
    # Step 2: Embed documents and query
    embedder = Embedder()
    doc_embeddings = embedder.embed_texts(docs)  # np.ndarray

    # Step 3: Build vector store from document embeddings
    dimension = doc_embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add_documents(doc_embeddings, docs)

    # Step 4: Search for similar documents based on query embedding
    query_embedding = embedder.embed_texts([query])[0]  # 1D np.ndarray
    retrieved_docs = vector_store.search(query_embedding, k=1)
    
    # Step 5: Generate answer conditioned on retrieved documents and query
    generator = AnswerGenerator()
    context = [doc["text"] for doc in retrieved_docs]
    answer = generator.generate_answer(query, context)
    
    return answer
