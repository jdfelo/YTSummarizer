#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import shutil
from openai import OpenAI
import whisper
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
import torch
import threading
import time
from tqdm import tqdm
import whisper.transcribe

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


class WhisperProgressBar(tqdm):
    """Custom progress bar for Whisper transcription"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_text = ""

    def set_description_str(self, desc=None, refresh=True):
        """Override to capture the current text being transcribed"""
        if desc and desc.startswith("Detected language:"):
            print(f"\n  [*] {desc}")
        elif desc and not desc.startswith("Transcribe"):
            self.current_text = desc
            # Clear the current line and print the text
            print(f"\r  [*] Transcribing: {desc}", end="\n", flush=True)
        super().set_description_str(desc, refresh)


def download_audio(youtube_url, output_base):
    """
    Download the audio from a YouTube URL and convert it to WAV.
    output_base is the base file path (without extension).
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        # Using a template that results in a final file like "audio.wav"
        'outtmpl': output_base + ".%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


def run_with_progress_bar(func, *args, **kwargs):
    """
    Runs a function in a separate thread while showing an indeterminate progress bar.
    Returns the function's result once done.
    """
    result_container = [None]
    done_event = threading.Event()

    def target():
        result_container[0] = func(*args, **kwargs)
        done_event.set()

    thread = threading.Thread(target=target)
    thread.start()

    # Create a progress bar that continuously updates until done.
    with tqdm(total=100, desc="Transcribing", leave=False, bar_format="{desc} {bar}") as pbar:
        while not done_event.is_set():
            time.sleep(0.5)
            pbar.update(1)
            if pbar.n >= 100:
                pbar.n = 0
                pbar.refresh()
        thread.join()
    return result_container[0]


def transcribe_audio(model, audio_path):
    """
    Transcribe the audio using Whisper with real-time progress tracking.
    """
    print("  [*] Running Whisper transcription...")
    
    # Create a progress bar for the transcription
    with tqdm(total=100, desc="  Progress", bar_format='{desc} {percentage:3.0f}% |{bar:30}| {elapsed}s') as pbar:
        def progress_callback(current, total):
            pbar.n = int(current * 100 / total)
            pbar.refresh()
            
        result = model.transcribe(
            audio_path,
            verbose=True,  # Enable verbose output
            condition_on_previous_text=True,
            initial_prompt="The following is a transcript of a video: "
        )
        
        # Ensure the progress bar reaches 100%
        pbar.n = 100
        pbar.refresh()
    
    return result['text']


def summarize_text(transcript):
    """
    Use GPT-4 to create a structured summary of police scanner transcripts.
    """
    print("  [*] Creating structured summary with GPT-4...")
    
    system_prompt = """You are a police dispatch analyst who creates clear, structured summaries of police scanner transcripts. 
    Organize the summary into these sections:
    1. Key Moments Summary - List major incidents and their details
    2. Suspicious Activity and Disturbances
    3. Theft and Burglary Incidents
    4. Juvenile and Welfare Checks
    5. Vehicle-Related Incidents
    6. Assault and Domestic Violence
    7. Public Safety Threats
    8. Firearms and Arrests
    9. Routine Reports and Cleanup Efforts
    10. Coordinated Responses and Emergency Services
    11. Key Locations - List all addresses mentioned with their associated incidents
    
    Format each section clearly with headers. Include specific details like addresses, descriptions, and outcomes when available.
    Focus on factual information and maintain a professional tone. List addresses in a structured format at the end."""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a structured summary of this police scanner transcript:\n\n{transcript}"}
        ],
        temperature=0.3,
        max_tokens=2000  # Increased token limit for detailed summaries
    )
    
    summary = completion.choices[0].message.content.strip()
    return summary


def save_transcript(text, output_path):
    """
    Save the transcript to a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [*] Transcript saved to: {output_path}")


def process_video(model, link, output_dir, mode="all"):
    """
    Process a single video with the specified mode.
    mode can be: "transcribe", "summarize", or "all"
    """
    # Create base filename from video ID
    video_id = link.split("watch?v=")[-1].split("&")[0]
    base_name = os.path.join(output_dir, video_id)
    
    # Create temporary directory for audio
    temp_dir = tempfile.mkdtemp(prefix="yt_transcribe_")
    temp_audio_base = os.path.join(temp_dir, "audio")
    
    try:
        print(f"\nProcessing video: {link}")
        
        # Download audio
        try:
            print("  [*] Downloading audio from YouTube...")
            download_audio(link, temp_audio_base)
        except Exception as e:
            print(f"  [!] Error downloading audio: {e}", file=sys.stderr)
            return

        # Check audio file
        temp_audio_file = temp_audio_base + ".wav"
        if not os.path.exists(temp_audio_file):
            print(f"  [!] Audio file not found: {temp_audio_file}", file=sys.stderr)
            return
        print(f"  [*] Found audio file: {temp_audio_file}")

        # Transcribe if needed
        if mode in ["transcribe", "all"]:
            try:
                transcript = transcribe_audio(model, temp_audio_file)
                # Save transcript
                transcript_file = f"{base_name}_transcript.txt"
                save_transcript(transcript, transcript_file)
            except Exception as e:
                print(f"  [!] Error during transcription: {e}", file=sys.stderr)
                return

        # Summarize if needed
        if mode in ["summarize", "all"] and 'transcript' in locals():
            try:
                summary = summarize_text(transcript)
                # Save summary
                summary_file = f"{base_name}_summary.txt"
                save_transcript(summary, summary_file)
            except Exception as e:
                print(f"  [!] Error during summarization: {e}", file=sys.stderr)
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe one or more YouTube videos using Whisper (with CUDA support if available) and summarize with GPT-4."
    )
    parser.add_argument('links', nargs='+', help='One or more YouTube video links.')
    parser.add_argument(
        '--api-key', type=str,
        help='Your OpenAI API key (alternatively, set the OPENAI_API_KEY variable in your .env file).'
    )
    parser.add_argument(
        '--mode', type=str, choices=['transcribe', 'summarize', 'all'],
        default='all', help='Operation mode: transcribe, summarize, or all (default: all)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory to save output files (default: output)'
    )
    args = parser.parse_args()

    # Set the OpenAI API key from command-line argument or environment variable.
    if args.mode in ['summarize', 'all']:
        if args.api_key:
            client.api_key = args.api_key
        elif os.getenv("OPENAI_API_KEY"):
            client.api_key = os.getenv("OPENAI_API_KEY")
        else:
            print("Error: Please provide an OpenAI API key either via --api-key or in the .env file (OPENAI_API_KEY).",
                  file=sys.stderr)
            sys.exit(1)

    # Check CUDA availability and initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        print(f"[*] Using GPU: {torch.cuda.get_device_name(0)}")
        torch.set_default_device(device)
    else:
        print("Warning: CUDA is not available. This will significantly slow down transcription.", file=sys.stderr)

    # Load the Whisper large model with progress bar
    print("[*] Loading Whisper large model...")
    with tqdm(total=100, desc="  Loading model", bar_format='{desc} {percentage:3.0f}% |{bar:30}| {elapsed}s') as pbar:
        pbar.n = 0
        pbar.refresh()
        model = whisper.load_model("large")
        pbar.n = 50
        pbar.refresh()
        model = model.to(device)
        pbar.n = 100
        pbar.refresh()

    # Process each video
    for link in args.links:
        process_video(model, link, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
