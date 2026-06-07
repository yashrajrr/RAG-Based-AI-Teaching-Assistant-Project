from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load once
df = joblib.load("dataframe.joblib")

stored_embeddings = np.vstack(df.embedding.values)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

"""
def create_embedding(texts):
    if not texts:
        return []

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    try:
        embedding = client.embeddings.create(
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            input=texts,
            encoding_format="float"
        )
    except Exception:
        return embedding_model.encode(texts)

    data = getattr(embedding, "data", None)
    if data:
        embeddings = []
        for item in data:
            if isinstance(item, dict):
                embeddings.append(item.get("embedding"))
            else:
                embeddings.append(getattr(item, "embedding", None))

        if all(e is not None for e in embeddings):
            return embeddings

    return embedding_model.encode(texts)
"""
def create_embedding(texts):
    if not texts:
        return []

    return embedding_model.encode(texts)

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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if not response.choices:
        return "No response generated."

    return response.choices[0].message.content


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
    Generate exactly 5 MCQs from all provided course chunks.

    Return two sections:

    STUDENT_QUIZ:
    - Show only question and options.
    - Do not show correct answers.
    - Do not show explanations.

    ANSWER_KEY:
    - For each question, include correct option.
    - Include short explanation.
    - Include source video and timestamp.

    Use all provided chunks across the quiz. Do not create all questions from only one chunk unless the topic appears only there.

    Each question should be based on a different idea from the retrieved chunks when possible.
    """

    return inference(prompt)

def create_quiz(topic) :
    chunks = retrieve_chunks(topic, top_results=10)
    quiz_output = generate_quiz(topic, chunks)
    return quiz_output, chunks


def ask_question(question):

    if not question.strip():
        return "Please enter a question.", []

    chunks = retrieve_chunks(question)

    answer = generate_response(
        question,
        chunks
    )

    return answer, chunks
