from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = "Answer ONLY using provided sources. If unsure, say you don't know."


def generate_answer(question, sources):
    context = "\n\n".join([s['text'][:500] for s in sources])

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
