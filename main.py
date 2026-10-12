import math
import os
import queue
import re
import shutil
import sys
import threading
from pathlib import Path

import ffmpeg
import pysrt
from faster_whisper import WhisperModel


def emit(log, message):
    print(message)
    if log is not None and log is not print:
        log(message)


def configure_ffmpeg():
    if shutil.which("ffmpeg"):
        return

    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            roots.append(Path(meipass))
    roots.append(Path(__file__).resolve().parent)

    for root in roots:
        matches = list(root.glob("ffmpeg/**/bin/ffmpeg.exe"))
        if not matches and (root / "ffmpeg.exe").exists():
            matches = [root / "ffmpeg.exe"]
        if matches:
            os.environ["PATH"] = str(matches[0].parent) + os.pathsep + os.environ.get("PATH", "")
            return

    raise FileNotFoundError(
        "ffmpeg was not found. Install ffmpeg and add its bin folder to PATH, "
        "or keep the ffmpeg folder next to this program."
    )


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

def extract_audio(video_path):
    video_path = Path(video_path)
    extracted_audio = video_path.with_name(f"audio-{video_path.stem}.wav")
    stream = ffmpeg.input(str(video_path))
    stream = ffmpeg.output(stream, str(extracted_audio))
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    return extracted_audio

def transcribe(audio, log=print):
    emit(log, "Loading the faster transcription model...")
    model = WhisperModel("turbo", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio),
        beam_size=1,
        vad_filter=True,
    )
    
    language = info.language
    emit(log, "Transcription language: " + language)
    
    if segments is None:
        emit(log, "No segments returned from transcription.")
        return language, []
    
    try:
        segments = list(segments)
    except Exception as e:
        emit(log, "Error converting segments to list: " + str(e))
        return language, []
    
    for segment in segments:
        emit(log, "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
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

def generate_subtitle_file(video_path, language, segments):
    video_path = Path(video_path)
    subtitle_file = video_path.with_name(f"sub-{video_path.stem}.{language}.srt")
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
    if token_count < 2:
        return text.strip()

    max_length = min(480, max(24, int(token_count * 0.8)))
    min_length = min(160, max(24, max_length // 2))

    if max_length >= token_count:
        max_length = max(8, token_count - 1)
    if min_length >= max_length:
        min_length = max(1, max_length // 2)

    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"].strip()

def summarize_sections(summarizer, text):
    chunks = chunk_transcript(text, summarizer.tokenizer, 450)
    summaries = [summarize_piece(summarizer, chunk) for chunk in chunks]
    return "\n\n".join(summary for summary in summaries if summary)

def summarize_text(text, log=print):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    emit(log, "Loading summary model...")
    summarizer = pipeline(
        "summarization",
        model="knkarthick/MEETING_SUMMARY",
        device=device,
    )
    emit(log, "Writing the long summary...")
    return summarize_sections(summarizer, text)

def generate_summary_file(video_path, language, segments, log=print):
    video_path = Path(video_path)
    summary_file = video_path.with_name(f"summary-{video_path.stem}.{language}.txt")
    transcript = transcript_text(segments)
    if not transcript:
        summary = "No speech was detected, so there is nothing to summarize.\n"
    else:
        long_summary = summarize_text(transcript, log=log)
        summary = "Long summary\n\n" + long_summary.strip() + "\n"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    emit(log, "Summary written to " + str(summary_file))
    emit(log, summary.rstrip())
    return summary_file

def run(video_path, log=print):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError("Video file not found: " + str(video_path))

    configure_ffmpeg()
    emit(log, "Selected file: " + str(video_path))
    emit(log, "Extracting audio...")
    extracted_audio = extract_audio(video_path)
    emit(log, "Transcribing...")
    language, segments = transcribe(audio=extracted_audio, log=log)
    subtitle_file = generate_subtitle_file(
        video_path=video_path,
        language=language,
        segments=segments
    )
    emit(log, "Subtitles written to " + str(subtitle_file))
    summary_file = generate_summary_file(
        video_path=video_path,
        language=language,
        segments=segments,
        log=log,
    )
    emit(log, "Finished.")
    return subtitle_file, summary_file


def launch_ui(initial_path=""):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class CaptionApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Generate Caption")
            self.geometry("760x520")
            self.minsize(640, 420)
            self.messages = queue.Queue()
            self.worker = None

            self.columnconfigure(0, weight=1)
            self.rowconfigure(2, weight=1)

            file_row = ttk.Frame(self, padding=(16, 16, 16, 8))
            file_row.grid(row=0, column=0, sticky="ew")
            file_row.columnconfigure(0, weight=1)

            self.path_var = tk.StringVar(value=initial_path)
            self.path_entry = ttk.Entry(file_row, textvariable=self.path_var)
            self.path_entry.grid(row=0, column=0, sticky="ew")
            ttk.Button(file_row, text="Browse...", command=self.browse).grid(row=0, column=1, padx=(8, 0))

            action_row = ttk.Frame(self, padding=(16, 0, 16, 8))
            action_row.grid(row=1, column=0, sticky="ew")
            self.generate_button = ttk.Button(action_row, text="Generate", command=self.start)
            self.generate_button.grid(row=0, column=0)
            self.progress = ttk.Progressbar(action_row, mode="indeterminate")
            self.progress.grid(row=0, column=1, sticky="ew", padx=(12, 0))
            action_row.columnconfigure(1, weight=1)

            log_frame = ttk.Frame(self, padding=(16, 0, 16, 16))
            log_frame.grid(row=2, column=0, sticky="nsew")
            log_frame.rowconfigure(0, weight=1)
            log_frame.columnconfigure(0, weight=1)
            self.log_box = tk.Text(log_frame, wrap="word", state="disabled")
            self.log_box.grid(row=0, column=0, sticky="nsew")
            scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
            scroll.grid(row=0, column=1, sticky="ns")
            self.log_box.configure(yscrollcommand=scroll.set)

            self.after(200, self.poll_messages)
            self.protocol("WM_DELETE_WINDOW", self.on_close)

        def browse(self):
            selected = filedialog.askopenfilename(
                title="Select a video",
                filetypes=[
                    ("MP4 video", "*.mp4"),
                    ("Video files", "*.mp4 *.mov *.mkv *.avi"),
                    ("All files", "*.*"),
                ],
            )
            if selected:
                self.path_var.set(selected)

        def append_log(self, message):
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def start(self):
            video_path = self.path_var.get().strip()
            if not video_path:
                messagebox.showwarning("Select a video", "Choose a video file first.")
                return
            if self.worker and self.worker.is_alive():
                return

            self.generate_button.state(["disabled"])
            self.progress.start(12)
            self.append_log("Starting...")
            self.worker = threading.Thread(
                target=self.run_job,
                args=(video_path,),
                daemon=True,
            )
            self.worker.start()

        def run_job(self, video_path):
            try:
                run(video_path, log=lambda message: self.messages.put(("log", message)))
                self.messages.put(("done", "Finished."))
            except ffmpeg.Error as exc:
                detail = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
                self.messages.put(("error", detail))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        def poll_messages(self):
            try:
                while True:
                    kind, message = self.messages.get_nowait()
                    if kind == "log":
                        self.append_log(message)
                    elif kind == "done":
                        self.progress.stop()
                        self.generate_button.state(["!disabled"])
                        self.append_log(message)
                        messagebox.showinfo(
                            "Generate Caption",
                            "The subtitle file and long summary are ready.",
                        )
                    elif kind == "error":
                        self.progress.stop()
                        self.generate_button.state(["!disabled"])
                        self.append_log(message)
                        messagebox.showerror("Generate Caption", message[:1200])
            except queue.Empty:
                pass
            self.after(200, self.poll_messages)

        def on_close(self):
            self.destroy()

    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    app = CaptionApp()
    app.mainloop()


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else ""
    launch_ui(initial)
