import os
import json
import shutil

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from chat_history import (
    add_message,
    clear_chat_history,
    create_session,
    get_session,
    load_chat_history,
    update_quiz_submission,
)
from get_output import ask_question, clean_output, create_quiz, get_topic_tags, reload_dataframe
import process_audio
import process_data
import process_json
import process_video

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "memora-dev-secret-key")
app.config["UPLOAD_FOLDER"] = "videos"
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "avi"}
COURSES_PATH = "courses.json"


def build_sources(chunks):
    sources = []
    for chunk in chunks:
        source = f"{chunk['video_name']} {chunk['start']}-{chunk['end']}"
        if source not in sources:
            sources.append(source)
    return sources


def get_ordered_sessions():
    sessions = load_chat_history()
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def get_video_names():
    if not os.path.isdir("clean_json_data"):
        return []

    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir("clean_json_data")
        if filename.endswith(".json")
    )


def course_id_from_name(name):
    safe_name = secure_filename(name).strip("._-").lower()
    return safe_name or "course"


def load_courses():
    if not os.path.exists(COURSES_PATH):
        return []

    try:
        with open(COURSES_PATH, "r", encoding="utf-8") as file:
            courses = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return courses if isinstance(courses, list) else []


def save_courses(courses):
    with open(COURSES_PATH, "w", encoding="utf-8") as file:
        json.dump(courses, file, indent=2)


def get_courses():
    courses = load_courses()
    assigned_videos = {
        video
        for course in courses
        for video in course.get("videos", [])
    }
    unassigned_videos = [
        video for video in get_video_names()
        if video not in assigned_videos
    ]

    if unassigned_videos:
        courses.append({
            "id": "imported-videos",
            "name": "Imported Videos",
            "description": "Videos processed before course folders were created.",
            "videos": unassigned_videos,
        })

    return courses


def add_video_to_course(video_name, course_dest, course_name, existing_course_id):
    courses = load_courses()

    if course_dest == "existing" and existing_course_id:
        course = next((item for item in courses if item.get("id") == existing_course_id), None)
    else:
        course = None

    if course is None:
        name = course_name.strip() or video_name
        base_id = course_id_from_name(name)
        course_id = base_id
        suffix = 2
        existing_ids = {item.get("id") for item in courses}
        while course_id in existing_ids:
            course_id = f"{base_id}-{suffix}"
            suffix += 1

        course = {
            "id": course_id,
            "name": name,
            "description": "Course folder for uploaded learning videos.",
            "videos": [],
        }
        courses.append(course)

    if video_name not in course["videos"]:
        course["videos"].append(video_name)

    save_courses(courses)


def remove_video_from_course(course_id, video_name):
    courses = load_courses()
    kept_courses = []

    for course in courses:
        if course.get("id") == course_id:
            course["videos"] = [
                video for video in course.get("videos", [])
                if video != video_name
            ]
        if course.get("videos"):
            kept_courses.append(course)

    save_courses(kept_courses)


def delete_video_files(video_name):
    candidates = [
        ("audios", f"{video_name}.mp3"),
        ("json_data", f"{video_name}.json"),
        ("clean_json_data", f"{video_name}.json"),
    ]

    if os.path.isdir("videos"):
        for filename in os.listdir("videos"):
            stem, extension = os.path.splitext(filename)
            if stem == video_name and extension.lower().lstrip(".") in ALLOWED_VIDEO_EXTENSIONS:
                candidates.append(("videos", filename))

    for folder, filename in candidates:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            os.remove(path)


def rebuild_dataframe_after_delete():
    has_clean_json = (
        os.path.isdir("clean_json_data")
        and any(filename.endswith(".json") for filename in os.listdir("clean_json_data"))
    )

    if has_clean_json:
        process_data.to_build_dataframe()
        reload_dataframe()
        return

    for path in ("dataframe.joblib", "processed_videos.joblib"):
        if os.path.exists(path):
            os.remove(path)
    reload_dataframe()


def merge_quiz_topic(user_input, selected_tags):
    parts = []
    for value in selected_tags + [user_input]:
        value = value.strip()
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def get_session_mode(session, requested_mode="qa"):
    if not session:
        return requested_mode or "qa"

    saved_mode = session.get("mode")
    if saved_mode:
        return saved_mode

    for message in session.get("messages", []):
        message_mode = message.get("mode")
        if message_mode:
            return message_mode

    return requested_mode or "qa"


def get_quiz_messages():
    quizzes = []
    for session in load_chat_history():
        for index, message in enumerate(session.get("messages", [])):
            if message.get("mode") == "quiz":
                quizzes.append({
                    "session": session,
                    "message": message,
                    "message_index": index
                })
    return quizzes


def build_quiz_report():
    quizzes = get_quiz_messages()
    completed = [item for item in quizzes if item["message"].get("quiz_submission")]
    strong_counts = {}
    weak_counts = {}

    for item in completed:
        message = item["message"]
        topic = message.get("question", "Quiz")
        for detail in message["quiz_submission"].get("details", []):
            target = strong_counts if detail.get("is_correct") else weak_counts
            target[topic] = target.get(topic, 0) + 1

    def ranked(counts):
        return [
            {"topic": topic, "count": count}
            for topic, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]

    total_score = sum(item["message"]["quiz_submission"].get("score", 0) for item in completed)
    total_questions = sum(item["message"]["quiz_submission"].get("total", 0) for item in completed)

    return {
        "quizzes": quizzes,
        "completed": completed,
        "total_score": total_score,
        "total_questions": total_questions,
        "strong_topics": ranked(strong_counts),
        "weak_topics": ranked(weak_counts),
    }


def process_uploaded_videos():
    os.makedirs("videos", exist_ok=True)
    os.makedirs("audios", exist_ok=True)
    os.makedirs("json_data", exist_ok=True)
    os.makedirs("clean_json_data", exist_ok=True)
    process_video.to_audio()
    process_audio.to_json()
    process_json.to_clean_json()
    process_data.to_build_dataframe()
    reload_dataframe()


def get_ffmpeg_exe():
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is not installed or available on PATH.")

    return ffmpeg_exe


def process_uploaded_video(filename, rebuild=True):
    os.makedirs("videos", exist_ok=True)
    os.makedirs("audios", exist_ok=True)
    os.makedirs("json_data", exist_ok=True)
    os.makedirs("clean_json_data", exist_ok=True)

    ffmpeg_exe = get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ.get("PATH", "")

    audio_files = process_video.to_audio(filename, ffmpeg_exe=ffmpeg_exe)
    if not audio_files:
        raise RuntimeError("The video was saved, but no audio file was created.")

    json_files = []
    for audio_file in audio_files:
        json_files.extend(process_audio.to_json(audio_file))

    clean_files = []
    for json_file in json_files:
        clean_files.extend(process_json.to_clean_json(json_file))

    if not clean_files:
        raise RuntimeError("The transcript was not created from the uploaded video.")

    if rebuild:
        process_data.to_build_dataframe()
        reload_dataframe()
    return clean_files


def process_uploaded_video_batch(filenames):
    processed = []
    for filename in filenames:
        process_uploaded_video(filename, rebuild=False)
        processed.append(os.path.splitext(filename)[0])

    process_data.to_build_dataframe()
    reload_dataframe()
    return processed


@app.route("/", methods=["GET", "POST"])
def index():
    return show_chat(None)


@app.route("/new-session")
def new_session():
    session = create_session()
    return redirect(url_for("show_chat", session_id=session["id"]))


@app.route("/chat/<session_id>", methods=["GET", "POST"])
def show_chat(session_id):
    session = get_session(session_id) if session_id else None
    mode = get_session_mode(session, request.form.get("mode", "qa"))

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        quiz_count = request.form.get("quiz_count", "5")
        video_name = request.form.get("video_name", "")
        selected_tags = request.form.getlist("topic_tags")

        if session is None:
            session = create_session(user_input[:40] or "New Chat")
            mode = request.form.get("mode", "qa")
        else:
            mode = get_session_mode(session, mode)

        if mode == "quiz":
            quiz_topic = merge_quiz_topic(user_input, selected_tags)
            try:
                quiz_count_value = int(quiz_count)
            except (TypeError, ValueError):
                quiz_count_value = 5
            result, answer_key, chunks, quiz_items = create_quiz(quiz_topic, quiz_count_value, video_name)
            result = clean_output(result)
            answer_key = clean_output(answer_key)
            sources = build_sources(chunks)
            quiz_meta = {
                "count": quiz_count_value,
                "video_name": video_name,
                "tags": selected_tags,
                "typed_topic": user_input
            }
            session = add_message(
                session["id"],
                quiz_topic,
                result,
                sources,
                "quiz",
                answer_key,
                quiz_items,
                quiz_meta,
            )
        else:
            result, chunks = ask_question(user_input)
            result = clean_output(result)
            sources = build_sources(chunks)
            session = add_message(session["id"], user_input, result, sources, "qa")

    sessions = get_ordered_sessions()
    return render_template(
        "index.html",
        mode=mode,
        session=session,
        sessions=sessions,
        videos=get_video_names(),
        topic_tags=get_topic_tags(),
        active_page="chat",
    )


@app.route("/videos", methods=["GET", "POST"])
def videos_page():
    if request.method == "POST":
        uploaded_videos = [
            video for video in request.files.getlist("video")
            if video and video.filename
        ]
        course_dest = request.form.get("course_dest", "new")
        course_name = request.form.get("course_name", "")
        existing_course_id = request.form.get("existing_course", "")

        if not uploaded_videos:
            flash("Please select at least one video file.", "error")
            return redirect(url_for("videos_page"))

        saved_filenames = []
        for video in uploaded_videos:
            if not video.filename or not allowed_video_file(video.filename):
                flash(f"{video.filename} is unsupported. Use mp4, webm, mov, mkv, or avi.", "error")
                continue

            filename = secure_filename(video.filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            video.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            saved_filenames.append(filename)

        if not saved_filenames:
            return redirect(url_for("videos_page"))

        try:
            video_names = process_uploaded_video_batch(saved_filenames)
            for video_name in video_names:
                add_video_to_course(video_name, course_dest, course_name, existing_course_id)

            count = len(video_names)
            label = "video" if count == 1 else "videos"
            flash(f"{count} {label} uploaded and processed successfully.", "success")
        except Exception as error:
            flash(f"Upload saved, but processing failed: {error}", "error")

        return redirect(url_for("videos_page"))

    return render_template(
        "videos.html",
        sessions=get_ordered_sessions(),
        videos=get_video_names(),
        courses=get_courses(),
        active_page="videos",
    )


@app.route("/courses/<course_id>/videos/<path:video_name>/delete", methods=["POST"])
def delete_course_video(course_id, video_name):
    remove_video_from_course(course_id, video_name)
    delete_video_files(video_name)

    try:
        rebuild_dataframe_after_delete()
        flash(f"{video_name} deleted successfully.", "success")
    except Exception as error:
        flash(f"{video_name} was deleted, but rebuilding search data failed: {error}", "error")

    return redirect(url_for("videos_page"))


@app.route("/quiz-report")
def quiz_report():
    return render_template(
        "quiz_report.html",
        sessions=get_ordered_sessions(),
        report=build_quiz_report(),
        active_page="report",
    )


@app.route("/chat/<session_id>/quiz/<int:message_index>/submit", methods=["POST"])
def submit_quiz(session_id, message_index):
    selected_answers = {}
    for key, value in request.form.items():
        if key.startswith("answer_"):
            selected_answers[key.replace("answer_", "", 1)] = value

    session = update_quiz_submission(session_id, message_index, selected_answers)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        if not session:
            return jsonify({"error": "Quiz not found."}), 404
        submission = session["messages"][message_index].get("quiz_submission", {})
        return jsonify({"submission": submission})

    return redirect(url_for("show_chat", session_id=session_id) + "#latest")


@app.route("/upload-video", methods=["POST"])
def upload_video():
    video = request.files.get("video")
    if not video or not video.filename:
        return redirect(url_for("index"))

    if allowed_video_file(video.filename):
        filename = secure_filename(video.filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        video.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        try:
            process_uploaded_video(filename)
            flash(f"{filename} uploaded and processed successfully.", "success")
        except Exception as error:
            flash(f"{filename} was uploaded, but processing failed: {error}", "error")

    return redirect(url_for("videos_page"))


@app.route("/clear-history", methods=["POST"])
def clear_history():
    clear_chat_history()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True, use_reloader=False)
