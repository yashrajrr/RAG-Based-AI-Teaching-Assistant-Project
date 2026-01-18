# RAG-Based AI Teaching Assistant for Educational Videos

## Project Description

This project is an AI-powered teaching assistant that processes educational videos to enable interactive question-answering based on the video content. It leverages Retrieval-Augmented Generation (RAG) techniques to convert videos into searchable knowledge bases, allowing users to ask questions and receive accurate, context-aware responses with references to specific video timestamps.

## Features

- **Video Processing Pipeline**: Automated conversion of videos to audio, transcription, and data cleaning.
- **Embedding-Based Search**: Uses sentence transformers to create embeddings for efficient similarity search.
- **Interactive Q&A**: Command-line interface for asking questions about video content.
- **Timestamp References**: Responses include relevant video timestamps for easy navigation.
- **Persistent Storage**: Processed data is cached to avoid reprocessing unchanged videos.

## Prerequisites

- Python 3.8 or higher
- `ffmpeg` installed and accessible in system PATH
- API keys for external services (see Setup section)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-ai-teaching-assistant
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `ffmpeg`:
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Setup

1. Create a `.env` file in the project root with your API keys:
   ```
   MIMO_API_KEY=your_openrouter_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here  # Currently unused, but loaded
   ```

2. Obtain API keys:
   - **OpenRouter API Key**: Sign up at [openrouter.ai](https://openrouter.ai) and get your API key. This is used for LLM inference.
   - **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey) (currently not used in the code).

## Usage

1. Place your educational videos in the `videos/` directory.

2. Run the main script to process videos and generate embeddings:
   ```bash
   python main.py
   ```

3. The script will:
   - Check if videos have been processed before.
   - If new or changed videos are detected, run the full pipeline.
   - Prompt for questions once processing is complete.

4. Enter your question when prompted and receive a response with video references.

5. Responses are also saved to `response.txt`.

## Project Structure

```
.
├── main.py                 # Main orchestrator script
├── video_tranformer.py     # Video to audio conversion
├── audio_transformer.py    # Audio transcription using Whisper
├── json_processor.py       # JSON cleaning and preprocessing
├── data_processor.py       # Embedding generation and dataframe creation
├── get_output.py           # Q&A interface and response generation
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── videos/                 # Input video files
├── audios/                 # Extracted audio files
├── json_data/              # Raw transcription JSON
├── clean_json_data/        # Processed JSON with chunks
├── dataframe.joblib        # Embedded data storage
├── processed_videos.joblib # List of processed videos
└── response.txt            # Latest Q&A response
```

## Workflow

1. **Video to Audio Conversion**
   - Videos are converted to MP3 using `ffmpeg`.

2. **Audio Transcription**
   - Audio files are transcribed to JSON using OpenAI's Whisper model (small variant).

3. **JSON Cleaning and Preprocessing**
   - Raw transcription segments are extracted and structured into chunks with timestamps.

4. **Embedding Generation**
   - Text chunks are embedded using Sentence Transformers (all-MiniLM-L6-v2).
   - Embeddings are stored in a pandas DataFrame.

5. **Question Answering**
   - User questions are embedded and compared to video content embeddings using cosine similarity.
   - Top similar chunks are retrieved and sent to an LLM for response generation.
   - Responses include video names and timestamp ranges.

## API Details

- **Transcription**: OpenAI Whisper (local, no API key required)
- **Embeddings**: Sentence Transformers (local)
- **LLM Inference**: OpenRouter API using Xiaomi MIMO v2 Flash model
- **Similarity Search**: Scikit-learn cosine similarity

## Output

- **Console**: Interactive Q&A with formatted responses
- **response.txt**: Latest answer saved to file
- **dataframe.joblib**: Persistent storage of embedded chunks
- **processed_videos.joblib**: Cache of processed video filenames

## Troubleshooting

- **ffmpeg not found**: Ensure ffmpeg is installed and in your system PATH.
- **API errors**: Check your API keys in `.env` file.
- **Processing fails**: Ensure videos are in supported formats (e.g., .webm, .mp4).
- **No response**: Verify that videos contain speech content and processing completed successfully.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Notes

- Currently supports English language videos.
- Optimized for educational content with clear speech.
- Processing time depends on video length and hardware.
- API usage may incur costs depending on your OpenRouter plan.

---

This project demonstrates the power of RAG techniques for educational content, enabling interactive learning experiences through AI-powered video analysis.
