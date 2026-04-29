from flask import Flask, render_template, request, jsonify
from services.qa_runtime import run_qa

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    result = run_qa(question)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
