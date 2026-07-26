import os
import joblib
import process_video 
import process_audio
import process_json
import process_data
from get_output import ask_question,create_quiz
from chat_history import save_chat

dataframe_path = 'dataframe.joblib'
processed_list_path = 'processed_videos.joblib'
videos_dir = 'videos'
video_extensions = {'.mp4', '.webm', '.mov', '.mkv', '.avi'}


def is_video_file(filename):
    return os.path.splitext(filename)[1].lower() in video_extensions

if not os.path.exists(dataframe_path) or not os.path.exists(processed_list_path):
    reprocess = True
else:
    current_videos = sorted([
        f for f in os.listdir(videos_dir)
        if os.path.isfile(os.path.join(videos_dir, f)) and is_video_file(f)
    ])
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

"""print("Choose Mode:")
print("1. Ask Question")
print("2. Generate Quiz")

choice = input("Enter choice: ")

sources = []

if choice == "1":
    question = input("Ask Your Question: ")
    answer, chunks = ask_question(question)
    if answer:
        refined_ans = answer.replace("**","")
        print(refined_ans)
        # code for chat history functionality
        for chunk in chunks:
                source = f"{chunk['video_name']} {chunk['start']}-{chunk['end']}"
                sources.append(source)
        save_chat(question, refined_ans, sources, mode="qa")

elif choice == "2":
    topic = input("Enter Quiz Topic: ")
    quiz, answer_key, chunks, _quiz_items = create_quiz(topic)
    if quiz:
        refined_quiz = quiz.replace("**","")
        print(refined_quiz)
        if answer_key:
            print("\nAnswer Key:\n")
            print(answer_key.replace("**",""))
        # code for chat history functionality
        for chunk in chunks:
                source = f"{chunk['video_name']} {chunk['start']}-{chunk['end']}"
                sources.append(source)
        save_chat(topic, f"{refined_quiz}\n\nAnswer Key:\n{answer_key}", sources, mode="quiz")

else:
    print("Invalid choice")"""



