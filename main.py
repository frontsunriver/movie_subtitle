import math
import re

import ffmpeg
import pysrt
from faster_whisper import WhisperModel

input_video = "input.mp4"
input_video_name = input_video.replace(".mp4", "")

def clean_text(text):
    # Remove extra spaces and fix common punctuation issues
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text

def translate_srt(input_file, output_file, target_language='zh-cn'):
    # Load the SRT file
    subs = pysrt.open(input_file)
    # Translate each subtitle
    for sub in subs:
        # Clean the text before translation
        cleaned_text = clean_text(sub.text)

        # Clean the translated text
        sub.text = cleaned_text

    # Save the translated subtitles to a new SRT file
    subs.save(output_file)

def extract_audio():
    extracted_audio = f"audio-{input_video_name}.wav"
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

def transcribe(audio):
    model = WhisperModel("large-v3")
    segments, info = model.transcribe(audio)
    
    language = info.language
    print("Transcription language:", language)
    
    if segments is None:
        print("No segments returned from transcription.")
        return language, []
    
    try:
        segments = list(segments)
    except Exception as e:
        print("Error converting segments to list:", e)
        return language, []
    
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
    
    return language, segments


def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time

def generate_subtitle_file(language, segments):
    subtitle_file = f"sub-{input_video_name}.{language}.srt"
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment.text} \n"
        text += "\n"
        
    # Open the file with UTF-8 encoding
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(text)

    return subtitle_file

def transcript_text(segments):
    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text and segment.text.strip()
    )

def chunk_transcript(text, tokenizer, max_tokens):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            flush()
            for start in range(0, len(token_ids), max_tokens):
                piece = tokenizer.decode(
                    token_ids[start:start + max_tokens],
                    skip_special_tokens=True,
                ).strip()
                if piece:
                    chunks.append(piece)
            continue
        if current and current_tokens + len(token_ids) > max_tokens:
            flush()
        current.append(sentence)
        current_tokens += len(token_ids)

    flush()
    return chunks

def summarize_piece(summarizer, text):
    token_count = len(summarizer.tokenizer.encode(text, add_special_tokens=False))
    max_length = min(180, max(24, token_count // 2))
    min_length = min(30, max(8, max_length // 4))
    if min_length >= max_length:
        min_length = max(1, max_length - 1)

    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"].strip()

def summarize_text(text):
    import torch
    from transformers import pipeline

    max_tokens = 900
    device = 0 if torch.cuda.is_available() else -1
    print("Loading summary model...")
    summarizer = pipeline(
        "summarization",
        model="knkarthick/MEETING_SUMMARY",
        device=device,
    )
    chunks = chunk_transcript(text, summarizer.tokenizer, max_tokens)
    summaries = [summarize_piece(summarizer, chunk) for chunk in chunks]
    return "\n\n".join(summary for summary in summaries if summary)

def generate_summary_file(language, segments):
    summary_file = f"summary-{input_video_name}.{language}.txt"
    transcript = transcript_text(segments)
    if not transcript:
        summary = "No speech was detected, so there is nothing to summarize."
    else:
        summary = summarize_text(transcript)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print("Summary written to", summary_file)
    print(summary)
    return summary_file

def add_subtitle_to_video(soft_subtitle, subtitle_file,  subtitle_language):

    video_input_stream = ffmpeg.input(input_video)
    subtitle_input_stream = ffmpeg.input(subtitle_file)
    output_video = f"output-{input_video_name}.mp4"
    subtitle_track_title = subtitle_file.replace(".srt", "")

    if soft_subtitle:
        stream = ffmpeg.output(
            video_input_stream, subtitle_input_stream, output_video, **{"c": "copy", "c:s": "mov_text"},
            **{"metadata:s:s:0": f"language={subtitle_language}",
            "metadata:s:s:0": f"title={subtitle_track_title}"}
        )
        ffmpeg.run(stream, overwrite_output=True)
    else :
        stream = ffmpeg.output(video_input_stream, output_video,

                               vf=f"subtitles={subtitle_file}")

        ffmpeg.run(stream, overwrite_output=True)

def run():
    extracted_audio = extract_audio()
    language, segments = transcribe(audio=extracted_audio)
    subtitle_file = generate_subtitle_file(
        language=language,
        segments=segments
    )
    generate_summary_file(
        language=language,
        segments=segments
    )

    add_subtitle_to_video(
        soft_subtitle=True,
        subtitle_file=subtitle_file,
        subtitle_language=language
    )

if __name__ == "__main__":
    run()