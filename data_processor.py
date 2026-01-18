import os 
import json
# import requests
import pandas as pd
import joblib
# from google import genai
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
import cohere



load_dotenv()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

def to_df():
    
    def create_embedding(text_list):
        """
        response = requests.post("http://localhost:11434/api/embed",json={
            "model" : "bge-m3",
            "input": text_list
        })
        """
        """
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents= text_list
        )
        
        embedding = response.embeddings
        return embedding
        """
        """
        embeddings = model.encode(text_list)
        return embeddings
        """
        embedding = co.embed(
            inputs=text_list,
            model="embed-v4.0",
            input_type="classification",
            embedding_types=["float"],
        )
        return embedding

    clean_json_files = os.listdir("clean_json_data")
    records =[]
    chunk_id = 0
    for file in clean_json_files:
        with open(f"clean_json_data/{file}") as f:
            json_data = json.load(f)
        print("\n"*1,"Embedding for file : ",file)
        embeddings = create_embedding([data['text'] for data in json_data['chunks']])
        
        for i,chunk in enumerate(json_data['chunks']):
            chunk['id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id+=1
            records.append(chunk)
        
        print(f"Done with Embedding file {file}","\n"*1)

       

    df = pd.DataFrame.from_records(records)
    joblib.dump(df,'dataframe.joblib')
    
    videos_dir = 'videos'
    video_files = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]
    joblib.dump(video_files, 'processed_videos.joblib')
