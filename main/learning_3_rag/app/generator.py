# Text generation using local LLM or API
from transformers import pipeline
from typing import List

class AnswerGenerator:
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100):
        self.generator = pipeline("text-generation", model=model_name, device="mps")
        self.max_new_tokens = max_new_tokens

    def generate_answer(self, query: str, context_documents: List[str]) -> str:
        # Concatenate context into one string
        context = "\n".join(context_documents)
        
        # Construct the prompt
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        
        # Generate answer from the LLM
        result = self.generator(prompt, max_new_tokens=self.max_new_tokens)
        
        return result[0]["generated_text"][len(prompt):].strip()
