# In this Python Script, We will be converting the videos to audios 
import os
import subprocess

videos_list = os.listdir("videos")
for vid in videos_list:
    curr_inp = os.path.join('videos',vid)
    name = vid.split(" [")[0]
    curr_op = os.path.join('audios',f"{name}.mp3")
    subprocess.run(['ffmpeg',"-i",curr_inp,"-vn","-acodec", "libmp3lame", curr_op])

