import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib
import json

def get_response():
    def create_embedding(text_list):
        response = requests.post("http://localhost:11434/api/embed",json={
            "model" : "bge-m3",
            "input": text_list
        })

        embedding = response.json()["embeddings"]
        return embedding

    def inference(prompt: str) -> str:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": True},
            stream=True
        ) as r:
            output = []
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if "response" in chunk:
                        text = chunk["response"]
                        print(text, end="", flush=True)  
                        output.append(text)
            print()  
            return "".join(output)


    df = joblib.load('dataframe.joblib')
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
    with open("response.txt",'w') as f:
        f.write(response)
        
    print(response)

