# main.py
"""
Entry point for the RAG demo project.
"""

from app.orchestration import orchestration

def main() -> None:
    """
    Main interactive entry for the RAG demo.
    Prompts user for a query and prints the generated answer.
    """
    query: str = input("📨 Please enter your question: ")

    print("🔍 Running RAG pipeline...")
    answer: str = orchestration(query)
    print("\n💬 Answer:")
    print(answer)
    
if __name__ == "__main__":
    main()
