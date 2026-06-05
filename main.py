import os
import joblib
import process_video 
import process_audio
import process_json
import process_data
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
    process_video.to_audio()

    print("\n"*2,"Converting Audios to JSON data","\n"*1)
    process_audio.to_json()

    print("\n"*2,"Preprocessing JSON data","\n"*1)
    process_json.to_clean_json()

    print("\n"*2,"Performing embeddings and saving in dataframe","\n"*1)
    process_data.to_build_dataframe()
else:
    print("\n"*2,"Dataframe is up to date, skipping processing steps.","\n"*1)

get_output.get_response()
