import video_tranformer 
import audio_transformer
import json_processor
import data_processor
import get_output


print("Processing Videos")
video_tranformer.to_audio()

print("Converting Audios to JSON data")
audio_transformer.to_json()

print("Preprocessing Json data")
json_processor.cleaning_json()

print("Performing embeddings and saving in dataframe")
data_processor.to_df()

print("\n"*100)
get_output.get_response()

