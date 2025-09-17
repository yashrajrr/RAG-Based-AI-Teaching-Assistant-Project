import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib

def create_embedding(text_list):
    response = requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input": text_list
    })

    embedding = response.json()["embeddings"]
    return embedding


df = joblib.load('dataframe.joblib')
question = input("Ask Your Question : ")
question_embedding = create_embedding([question])[0]


print("print df embedding content")

# performing similarity with normal embedding and cosine embedding
similarity = cosine_similarity(np.vstack(df.embedding.values),[question_embedding]).flatten()

top_results = 15
max_indices = similarity.argsort()[::-1][0:top_results]

new_df = df.loc[max_indices]
# print(new_df[['video_name','id',"text"]])

for index,item in new_df.iterrows():
    print(index , item['video_name'] , item['text'] , item['start'] , item['end'])
