import os

import joblib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Load once
df = joblib.load("dataframe.joblib")

stored_embeddings = np.vstack(df.embedding.values)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def create_embedding(texts):
    if not texts:
        return []

    return embedding_model.encode(texts, convert_to_numpy=True)

def format_timestamp(seconds):
    seconds = float(seconds)
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def retrieve_chunks(question, top_results=5):
    """
    Returns top matching subtitle chunks.
    """

    question_embedding = np.array(create_embedding([question])[0])

    similarity = cosine_similarity(
        stored_embeddings,
        question_embedding.reshape(1, -1)
    ).flatten()
    
    max_indices = similarity.argsort()[::-1][:top_results]

    results_df = df.loc[max_indices].copy()
    results_df["start"] = results_df["start"].apply(
        format_timestamp
    )
    results_df["end"] = results_df["end"].apply(
        format_timestamp
    )

    results_df["similarity"] = similarity[max_indices]
    chunks = results_df[
        [
            "video_name",
            "text",
            "start",
            "end",
            "similarity"
        ]
    ].to_dict(orient="records")


    return chunks


def inference(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key is missing. Please add OPENROUTER_API_KEY to your .env file."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as error:
        return f"Could not generate a response: {error}"

    if not response.choices:
        return "No response generated. The selected model may be unavailable. Try changing OPENROUTER_MODEL in .env."

    message = response.choices[0].message.content
    if not message:
        return "The model returned an empty response. Try another OpenRouter model."

    return message


def generate_response(question, chunks):
    """
    Sends context to LLM and returns answer.
    """
    prompt = f"""
    You are an assistant helping students learn from videos.

    Here are subtitle chunks:
    {chunks}

    User question:
    "{question}"

    Instructions:
    1. If the question relates to the course content:
       - Never mention JSON.
       - Identify the relevant video(s).
       - Provide timestamp range.
       - Give a short explanation.
       - Only answer using the provided subtitle chunks.

    2. If unrelated, politely refuse.

    3. Professional formatting.

    4. Do not use markdown bold.
    """

    response = inference(prompt)

    if response is None:
        response = "No response generated. Please try again."

    return response

def generate_quiz(topic, chunks):
    prompt = f"""
    You are an AI teaching assistant creating a quiz from course video transcripts.

    Topic:
    "{topic}"

    Course chunks:
    {chunks}

    Generate exactly 5 multiple-choice questions from the provided course chunks.

    Return two sections:

    STUDENT_QUIZ:
    - Show only the question and 4 options labeled A, B, C, and D.
    - Do not show correct answers.
    - Do not show explanations.

    ANSWER_KEY:
    - For each question, include correct option.
    - Include short explanation.
    - Include source video and timestamp.

    Use all provided chunks across the quiz. Do not create all questions from only one chunk unless the topic appears only there.
    Each question should be based on a different idea from the retrieved chunks when possible.
    If the topic is not covered in the provided chunks, say: "This topic is not clearly covered in the course videos."
    Do not invent facts outside the provided chunks.
    Do not mention JSON or raw chunks.
    """

    return inference(prompt)

def split_quiz_output(quiz_output):
    student_marker = "STUDENT_QUIZ:"
    answer_marker = "ANSWER_KEY:"

    if answer_marker not in quiz_output:
        return quiz_output.strip(), ""

    student_part = quiz_output
    if student_marker in quiz_output:
        student_part = quiz_output.split(student_marker, 1)[1]

    student_part, answer_part = student_part.split(answer_marker, 1)
    return student_part.strip(), answer_part.strip()


def create_quiz(topic):
    if not topic.strip():
        return "Please enter a quiz topic.", "", []

    chunks = retrieve_chunks(topic, top_results=10)
    quiz_output = generate_quiz(topic, chunks)
    student_quiz, answer_key = split_quiz_output(quiz_output)
    return student_quiz, answer_key, chunks


def ask_question(question):

    if not question.strip():
        return "Please enter a question.", []

    chunks = retrieve_chunks(question)

    answer = generate_response(
        question,
        chunks
    )

    return answer, chunks
