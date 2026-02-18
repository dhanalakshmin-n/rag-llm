import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize client (for Mistral/OpenRouter)
client = OpenAI(base_url="https://openrouter.ai/api/v1")


def build_prompt(query: str, context_chunks: list):
    """
    Build RAG prompt using retrieved context
    """

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{query}

Answer:
"""

    return prompt.strip()


def generate_answer(query: str, context_chunks: list):
    """
    Generate answer using Mistral
    """

    prompt = build_prompt(query, context_chunks)

    response = client.responses.create(
        model="mistralai/mistral-7b-instruct",
        input=prompt
    )

    return response.output_text
