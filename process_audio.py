import whisper
import os
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def is_audio_file(filename):
    return os.path.splitext(filename)[1].lower() in AUDIO_EXTENSIONS


def to_json(audio_filename=None):
    model_name = os.getenv("WHISPER_MODEL", "tiny")
    model = whisper.load_model(model_name)

    audios_files = [audio_filename] if audio_filename else os.listdir("audios")
    os.makedirs("json_data", exist_ok=True)
    created_files = []

    for file_name in audios_files:
        audio_path = os.path.join("audios", file_name)
        if not os.path.isfile(audio_path) or not is_audio_file(file_name):
            continue

        print("Processing file ",file_name)
        name = os.path.splitext(file_name)[0]
        result = model.transcribe(audio_path, fp16=False)
        
        output_name = f"{name}.json"
        with open(os.path.join("json_data", output_name), "w") as f:
            json.dump(result,f,indent = 4,default=str)
        created_files.append(output_name)
        
    print("\n"*2)
    return created_files
