from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

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
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
