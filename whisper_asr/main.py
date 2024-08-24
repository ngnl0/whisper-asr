import argparse
import glob
import os
import tkinter as tk
from tkinter import filedialog

import whisper
from whisper import Whisper
from loguru import logger


def find_files_with_suffix(root_folder, target_suffix) -> list[str]:
    """
    Find files with a specific suffix in a given root folder.

    Args:
        root_folder (str): The root folder to start the search from.
        target_suffix (str): The suffix of the files to search for.

    Returns:
        list: A list of file paths that match the given suffix.
    """
    pattern = os.path.join(root_folder, "**", f"*{target_suffix}")
    result_files = glob.glob(pattern, recursive=True)
    return result_files


def select_file() -> str:
    """
    Opens a file dialog to allow the user to select a file.
    Returns:
        str: The path of the selected file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path: str = filedialog.askopenfilename(title="Select a file")
    root.destroy()
    return file_path


def select_directory() -> str:
    """
    Opens a file dialog to select a folder and returns the selected folder path.
    Returns:
        str: The path of the selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory(title="Select a folder")
    root.destroy()
    return file_path


def transcribe_video(model: Whisper, srt_path: str) -> None:
    """
    Transcribes the given video using the specified model.
    Args:
        model: The model used for transcription.
        video_path: The path to the video file.
    Returns:
        None
    Raises:
        None
    """
    logger.info(f"Transcribing {srt_path}...")
    result: dict[str, str | list] = model.transcribe(srt_path, verbose=True)
    srt_path: str = os.path.splitext(srt_path)[0] + ".en.whisper.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"{segment['id'] + 1}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

    logger.success(f"Transcription successful for {srt_path}")


def format_timestamp(seconds) -> str:
    """
    Formats a timestamp in seconds to the following format: HH:MM:SS,mmm.

    Args:
        seconds (float): The timestamp in seconds.

    Returns:
        str: The formatted timestamp in the format HH:MM:SS,mmm.
    """
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def find_files_with_suffix_glob(directory: str):
    """
    Find files with specific suffixes in a directory and its subdirectories.
    Args:
        directory (str): The directory to search for files.
    Returns:
        list: A list of file paths that match the specified suffixes.
    """
    matching_files = []

    suffixes = ["*.m4a", "*.mp3", "*.webm", "*.mp4", "*.mpga", "*.wav", "*.mpeg"]
    for suffix in suffixes:
        pattern = os.path.join(directory, "**", f"*{suffix}")
        matching_files.extend(glob.glob(pattern, recursive=True))
    return matching_files


def transcribe_single_file(model: Whisper) -> None:
    selected_path = select_file()
    transcribe_video(model, selected_path)


def transcribe_multiple_files(model: Whisper) -> None:
    selected_path = select_directory()
    video_files_path = find_files_with_suffix_glob(selected_path)
    for video_file_path in video_files_path:
        transcribe_video(model, video_file_path)


def main():
    parser = argparse.ArgumentParser(description="Whisper Automatic Speech Recognition")
    parser.add_argument(
        "-v", "--version", help="show Whisper ASR version. ", action="store_true"
    )
    parser.add_argument(
        "-a",
        "--all-file",
        help="transcribe all files in select directory",
        action="store_true",
    )
    args = parser.parse_args()
    if args.version:
        print("Whisper ASR version: 0.1.0")
    elif args.all_file:
        logger.info("Initializing the model ...")
        model: Whisper = whisper.load_model("base")
        transcribe_multiple_files(model)
    else:
        logger.info("Initializing the model ...")
        model: Whisper = whisper.load_model("base")
        transcribe_single_file(model)


if __name__ == "__main__":
    main()
