# app/orchestration.py

from typing import List
from app.document_loader import load_documents_txt, split_documents
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
    # Step 1.5: Split documents into chunks
    chunks: List[str] = split_documents(docs, chunk_size=300, overlap=50)

    # Step 2: Embed document chunks and query
    embedder = Embedder()
    doc_embeddings = embedder.embed_texts(chunks)  # np.ndarray

    # Step 3: Build vector store from document embeddings
    dimension = doc_embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add_documents(doc_embeddings, chunks)

    # Step 4: Search for similar document chunks based on query embedding
    query_embedding = embedder.embed_texts([query])[0]  # 1D np.ndarray
    retrieved_docs = vector_store.search(query_embedding, k=3)

    # Step 5: Generate answer conditioned on retrieved document chunks and query
    generator = AnswerGenerator()
    context = [doc["text"] for doc in retrieved_docs]
    answer = generator.generate_answer(query, context)
    
    return answer
