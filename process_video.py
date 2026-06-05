# In this Python Script, We will be converting the videos to audios 
import os
import subprocess

def to_audio():
    videos_list = os.listdir("videos")
    for vid in videos_list:
        print("Processing video ", vid )
        curr_inp = os.path.join('videos',vid)
        name = vid.split(" [")[0]
        curr_op = os.path.join('audios',f"{name}.mp3")
        
        subprocess.run([
            "ffmpeg",
            "-y",                 
            "-hide_banner",        
            "-loglevel", "quiet",  
            "-i", curr_inp,
            "-vn",
            "-acodec", "libmp3lame",
            curr_op
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        


