from flask import Flask, request, render_template
from loadollama_loadvectordb_setupqa import ask_question

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        question = request.form["question"]
        answer = ask_question(question)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
