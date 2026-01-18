import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ---------- ENV & CLIENT ----------
load_dotenv()


# ---------- EMBEDDING ----------
def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    return embeddings


# ---------- MAIN PIPELINE ----------
def build_dataframe(
    json_dir="clean_json_data",
    videos_dir="videos",
    df_out="dataframe.joblib",
    videos_out="processed_videos.joblib",
):
    records = []
    chunk_id = 0

    for file in os.listdir(json_dir):
        path = os.path.join(json_dir, file)
        if not path.endswith(".json"):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        texts = [c.get("text", "").strip() for c in chunks if c.get("text")]

        if not texts:
            continue

        print(f"Embedding → {file}")
        embeddings = create_embeddings(texts)

        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding count mismatch")

        emb_idx = 0
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            record = dict(chunk)
            record["id"] = chunk_id
            record["embedding"] = embeddings[emb_idx]

            records.append(record)
            chunk_id += 1
            emb_idx += 1

    if not records:
        raise RuntimeError("No records generated")

    df = pd.DataFrame(records)
    joblib.dump(df, df_out)

    if os.path.isdir(videos_dir):
        videos = [
            f for f in os.listdir(videos_dir)
            if os.path.isfile(os.path.join(videos_dir, f))
        ]
        joblib.dump(videos, videos_out)

    print(f"Saved {len(df)} chunks")


# ---------- RUN ----------
if __name__ == "__main__":
    build_dataframe()
