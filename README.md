## Instructions to Run the YouTube Video Transcription and Summarization Script

### Prerequisites
1. **Python 3.7+** installed.
2. **Pip dependencies**: Install the necessary Python libraries:
   ```bash
   pip install torch whisper yt-dlp tqdm python-dotenv openai
   ```
3. **FFmpeg** installed (required for audio processing):
   - For Linux/macOS:
     ```bash
     sudo apt-get install ffmpeg  # For Debian-based systems
     brew install ffmpeg          # For macOS
     ```
   - For Windows: Download from [FFmpeg.org](https://ffmpeg.org/) and add to PATH.

4. **OpenAI API Key**: 
   - Create a `.env` file in the project directory and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
   - Alternatively, you can provide the API key as a command-line argument.

---

### How to Run the Script

1. **Clone or download the script**.
2. Open a terminal, navigate to the script directory, and run the following command:

```bash
python3 script_name.py <YouTube links> [OPTIONS]
```

### Example Usage:
```bash
python3 script_name.py "https://www.youtube.com/watch?v=example_video" --mode all
```

---

### Command-Line Options:
| **Argument**              | **Description**                                                                                   | **Default**      |
|--------------------------|---------------------------------------------------------------------------------------------------|------------------|
| `<YouTube links>`         | One or more YouTube links to process.                                                            | Required         |
| `--api-key`               | Your OpenAI API key (can be skipped if set in `.env`).                                           | None             |
| `--mode`                  | Mode of operation: `transcribe`, `summarize`, or `all`.                                          | `all`            |
| `--output-dir`            | Directory where output files (transcripts and summaries) will be saved.                         | `output`         |

---

### Modes of Operation:
- **transcribe**: Only transcribes the YouTube video using Whisper.
- **summarize**: Summarizes a pre-transcribed video using GPT-4.
- **all**: Transcribes the video and then generates a summary.

---

### Output Files:
- **Transcript**: Saved as `<video_id>_transcript.txt` in the output directory.
- **Summary**: Saved as `<video_id>_summary.txt` in the output directory (if summarization is selected).

---

### Example Output:
When run successfully, the script will output:
- Transcription progress.
- Summarization progress.
- Links to saved files in the specified directory.

### Troubleshooting:
1. **OpenAI API Key Issues**: Ensure the API key is correctly set in `.env` or provided via the `--api-key` argument.
2. **CUDA Warning**: If CUDA is not available, the script will run on the CPU, which may slow down transcription. For GPU users, ensure CUDA is properly installed.
