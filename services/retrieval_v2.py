import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None

DATA_DIR = "data"

def load_documents():
    docs = []
    for f in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, f), "r") as file:
            docs.append({"name": f, "text": file.read()})
    return docs


def embed(text):
    if client:
        return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    return np.random.rand(1536)


def retrieve(query):
    docs = load_documents()
    q_emb = embed(query)

    results = []
    for d in docs:
        d_emb = embed(d["text"][:1000])
        score = cosine_similarity([q_emb], [d_emb])[0][0]
        results.append((score, d))

    results.sort(reverse=True, key=lambda x: x[0])
    top = results[:3]

    return [{"score": float(s), "text": d["text"], "source": d["name"]} for s, d in top]
