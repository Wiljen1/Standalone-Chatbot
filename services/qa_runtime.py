from services.retrieval_v2 import retrieve
from services.llm_runtime import generate_answer
import json

LOG_FILE = "qa_log.json"

def confidence_score(results):
    if not results:
        return "Low"
    avg = sum(r['score'] for r in results) / len(results)
    if avg > 0.75:
        return "High"
    elif avg > 0.5:
        return "Medium"
    return "Low"


def log(entry):
    try:
        data = []
        try:
            with open(LOG_FILE, 'r') as f:
                data = json.load(f)
        except:
            pass
        data.append(entry)
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass


def run_qa(question):
    results = retrieve(question)
    answer = generate_answer(question, results)
    conf = confidence_score(results)

    entry = {
        "question": question,
        "answer": answer,
        "confidence": conf,
        "sources": [r['source'] for r in results]
    }

    if conf == "Low" or not results:
        log(entry)

    return {
        "answer": answer,
        "sources": results,
        "confidence": conf
    }
