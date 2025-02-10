#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import shutil
import openai
import whisper
from yt_dlp import YoutubeDL


def download_audio(youtube_url, output_path):
    """
    Download the audio from a YouTube URL and convert it to WAV.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,  # Output file template
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


def transcribe_audio(model, audio_path):
    """
    Transcribe the audio using Whisper.
    """
    print("  [*] Running Whisper transcription...")
    result = model.transcribe(audio_path)
    return result['text']


def summarize_text(transcript):
    """
    Use GPT-4 via OpenAI's ChatCompletion API to summarize the transcript.
    """
    print("  [*] Summarizing transcript with GPT-4...")
    prompt = f"Summarize the following transcript in a concise and clear manner:\n\n{transcript}"

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Change this if you have a different model/version available
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300,  # Adjust token count as needed
    )
    summary = response.choices[0].message.content.strip()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe one or more YouTube videos using Whisper and summarize with GPT-4."
    )
    parser.add_argument('links', nargs='+', help='One or more YouTube video links.')
    parser.add_argument(
        '--api-key', type=str,
        help='Your OpenAI API key (alternatively, set the OPENAI_API_KEY environment variable).'
    )
    args = parser.parse_args()

    # Set the OpenAI API key
    if args.api_key:
        openai.api_key = args.api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        print(
            "Error: Please provide an OpenAI API key either via --api-key or by setting the OPENAI_API_KEY environment variable.",
            file=sys.stderr)
        sys.exit(1)

    # Load the Whisper large model (this may take a while on the first run)
    print("[*] Loading the Whisper large model. This might take several minutes...")
    model = whisper.load_model("large")

    # Create a temporary directory for audio files
    temp_dir = tempfile.mkdtemp(prefix="yt_transcribe_")

    try:
        for link in args.links:
            print(f"\nProcessing video: {link}")
            # Generate a safe temporary filename for the audio output
            temp_audio_file = os.path.join(temp_dir, "audio.wav")

            try:
                print("  [*] Downloading audio from YouTube...")
                download_audio(link, temp_audio_file)
            except Exception as e:
                print(f"  [!] Error downloading audio: {e}", file=sys.stderr)
                continue

            # Transcribe the audio
            transcript = transcribe_audio(model, temp_audio_file)

            # Summarize the transcript using GPT-4
            summary = summarize_text(transcript)

            # Display the results
            print("\n--- Transcription Summary ---")
            print(summary)
            print("-----------------------------\n")
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
