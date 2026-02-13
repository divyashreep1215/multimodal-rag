import os
from groq import Groq

# Read API key from Streamlit Secrets or environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, context_docs):
    context = "\n\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # ✅ Hardcoded valid Groq model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content
