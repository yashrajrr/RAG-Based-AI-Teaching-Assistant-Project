import os 
import json
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    response = requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input": text_list
    })

    embedding = response.json()["embeddings"]
    return embedding

clean_json_files = os.listdir("clean_json_data")

records =[]
chunk_id = 0
for file in clean_json_files:
    with open(f"clean_json_data/{file}") as f:
        json_data = json.load(f)
        
    print("Embedding for file : ",file)
    embeddings = create_embedding([data['text'] for data in json_data['chunks']])
    
    for i,chunk in enumerate(json_data['chunks']):
        chunk['id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id+=1
        records.append(chunk)
    print(f"Done with Embedding file {file}")
# parsing to pandas
df = pd.DataFrame.from_records(records)
joblib.dump(df,'dataframe.joblib')

"""
question = input("Ask Your Question : ")
question_embedding = create_embedding([question])[0]

# print(question_embedding)

print("print df embedding content")

# print(np.vstack(df.embedding.values))
# print(np.vstack(df.embedding).shape)

# performing similarity with normal embedding and cosine embedding
similarity = cosine_similarity(np.vstack(df.embedding.values),[question_embedding]).flatten()
max_indices = similarity.argsort()[::-1][0:5]

print(max_indices)
for i in max_indices:
    # print(df[df['id'] == i]["text"])
    print(df.loc[i,'text'])

"""