import os
import joblib
import video_tranformer 
import audio_transformer
import json_processor
import data_processor
import get_output

dataframe_path = 'dataframe.joblib'
processed_list_path = 'processed_videos.joblib'
videos_dir = 'videos'

if not os.path.exists(dataframe_path) or not os.path.exists(processed_list_path):
    reprocess = True
else:
    current_videos = sorted([f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))])
    processed_videos = joblib.load(processed_list_path)
    reprocess = current_videos != processed_videos


if reprocess:
    print("Processing Videos","\n"*1)
    video_tranformer.to_audio()

    print("\n"*2,"Converting Audios to JSON data","\n"*1)
    audio_transformer.to_json()

    print("\n"*2,"Preprocessing JSON data","\n"*1)
    json_processor.cleaning_json()

    print("\n"*2,"Performing embeddings and saving in dataframe","\n"*1)
    data_processor.to_df()
else:
    print("\n"*2,"Dataframe is up to date, skipping processing steps.","\n"*1)

get_output.get_response()
