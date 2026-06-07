import os
import subprocess


def video_title(filename):
    stem = os.path.splitext(filename)[0]
    return stem.split(" [")[0]


def to_audio(video_filename=None, ffmpeg_exe="ffmpeg"):
    os.makedirs("audios", exist_ok=True)

    videos_list = [video_filename] if video_filename else os.listdir("videos")
    created_files = []

    for video in videos_list:
        input_path = os.path.join("videos", video)
        if not os.path.isfile(input_path):
            continue

        audio_name = f"{video_title(video)}.mp3"
        output_path = os.path.join("audios", audio_name)
        print("Processing video", video)

        result = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "libmp3lame",
                output_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Could not convert {video} to audio.")

        created_files.append(audio_name)

    return created_files
