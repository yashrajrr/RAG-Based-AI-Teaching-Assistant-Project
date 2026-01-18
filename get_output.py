from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = os.getenv('GEMINI_API_KEY')

def get_response():
    def create_embedding(texts):
        if not texts:
            return []
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings

    def inference(prompt: str) -> str:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("MIMO_API_KEY")
            )

            response = client.chat.completions.create(
                model="xiaomi/mimo-v2-flash:free",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                extra_body={"reasoning": {"enabled": True}}
            )

            return response.choices[0].message.content

    df = joblib.load('dataframe.joblib')
    print("\n"*5)
    question = input("Ask Your Question : ")
    question_embedding = create_embedding([question])[0]

    # performing similarity with normal embedding and cosine embedding
    similarity = cosine_similarity(np.vstack(df.embedding.values),[question_embedding]).flatten()

    top_results = 5
    max_indices = similarity.argsort()[::-1][0:top_results]

    new_df = df.loc[max_indices]

    def format_timestamp(seconds):
        seconds = float(seconds)
        minutes, secs = divmod(round(seconds), 60)  # round instead of int
        return f"{minutes:02d}:{secs:02d}"


    # Format start & end into mm:ss
    formatted_df = new_df.copy()
    formatted_df["start"] = formatted_df["start"].apply(format_timestamp)
    formatted_df["end"] = formatted_df["end"].apply(format_timestamp)

    # Convert to list of dicts for clean JSON
    subtitle_chunks = formatted_df[["video_name", "text", "start", "end"]].to_dict(orient="records")

    prompt = f"""
    You are an assistant helping students learn from videos.

    Here are subtitle chunks (JSON list):
    {subtitle_chunks}

    User question:
    "{question}"

    Instructions:
    1. If the question relates to the course content:
    - Never Mentioned the json just talk in natural human language
    - Identify which video(s) contain the answer.
    - Provide the timestamp range (start–end) in mm:ss format.
    - Give a short, helpful explanation guiding the student to that part of the video.

    2. If the question is unrelated to the course, politely say you can only answer questions about the course content.
    """

    response = inference(prompt)

    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)

    print(response)


