import os
import json


def cleaning_json():
    json_files = os.listdir('json_data')
    os.makedirs("clean_json_data",exist_ok=True)
    for file in json_files:
        clean_json = []
        name = file.split('.')[0]
        with open(f"json_data/{file}", "r") as f:
            data = json.load(f)
        curr_data = data["segments"]
        for itr in curr_data:
            
            clean_json.append({
                'video_name':name,
                'text' : itr['text'],
                'start' : f"{itr['start']:.2f}",
                'end' : f"{itr['end']:.2f}"
                })
            
        with open(f"clean_json_data/{file}", "w") as f:
            json.dump({"chunks":clean_json,"full_text":data["text"]},f,indent = 4)
        