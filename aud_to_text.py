import whisper
import os
import json

model = whisper.load_model("small")

audios_files = os.listdir("audios")
os.makedirs("json_data", exist_ok=True)

for file_name in audios_files:
    name = file_name.split('.')[0]
    result = model.transcribe(f"audios/{file_name}")
    
    with open(f'json_data/{name}.json','w') as f:
        json.dump(result,f,indent = 4,default=str)
    
    print("Done with audio : ",file_name)