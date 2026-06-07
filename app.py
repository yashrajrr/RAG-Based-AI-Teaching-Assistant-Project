from flask import Flask, redirect, render_template, request, url_for

from chat_history import clear_chat_history, load_chat_history, save_chat
from get_output import ask_question, create_quiz

app = Flask(__name__)


def build_sources(chunks):
    sources = []
    for chunk in chunks:
        source = f"{chunk['video_name']} {chunk['start']}-{chunk['end']}"
        if source not in sources:
            sources.append(source)
    return sources


@app.route("/", methods=["GET", "POST"])
def index():
    mode = request.form.get("mode", "qa")
    user_input = ""
    result = None
    answer_key = ""
    sources = []

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()

        if mode == "quiz":
            result, answer_key, chunks = create_quiz(user_input)
            sources = build_sources(chunks)
            save_text = result
            if answer_key:
                save_text = f"{result}\n\nAnswer Key:\n{answer_key}"
            save_chat(user_input, save_text, sources, mode="quiz")
        else:
            result, chunks = ask_question(user_input)
            sources = build_sources(chunks)
            save_chat(user_input, result, sources, mode="qa")

    history = load_chat_history()
    return render_template(
        "index.html",
        mode=mode,
        user_input=user_input,
        result=result,
        answer_key=answer_key,
        sources=sources,
        history=reversed(history[-8:]),
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    clear_chat_history()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
