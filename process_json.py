import os
import json


def to_clean_json(json_filename=None):
    json_files = [json_filename] if json_filename else os.listdir('json_data')
    os.makedirs("clean_json_data",exist_ok=True)
    created_files = []

    for file in json_files:
        path = os.path.join("json_data", file)
        if not os.path.isfile(path):
            continue

        clean_json = []
        name = os.path.splitext(file)[0]
        with open(path, "r") as f:
            data = json.load(f)
        curr_data = data["segments"]
        for itr in curr_data:
            
            clean_json.append({
                'video_name':name,
                'text' : itr['text'],
                'start' : f"{itr['start']:.2f}",
                'end' : f"{itr['end']:.2f}"
                })
            
        output_path = os.path.join("clean_json_data", file)
        with open(output_path, "w") as f:
            json.dump({"chunks":clean_json,"full_text":data["text"]},f,indent = 4)
        created_files.append(file)

    return created_files
        
