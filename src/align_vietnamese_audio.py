#!/usr/bin/env python3
"""
Minimal helpers for Whisper transcription and audio cutting.

This module is intentionally lean:
- run_whisper_transcription(): transcribe audio with Whisper and save segments/words.
- cut_audio_segments(): cut audio into per-sentence WAVs (keeps legacy add_padding kw).
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import librosa
import soundfile as sf
import whisper


def save_whisper_results(result: dict, output_dir: str) -> None:
    """Persist Whisper segments, words, and transcription."""
    os.makedirs(output_dir, exist_ok=True)
    
    segments = []
    words = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
        )
        for word in seg.get("words", []):
            words.append(
                {
                    "word": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                }
            )

    with open(os.path.join(output_dir, "whisper_segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "whisper_words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "whisper_transcription.txt"), "w", encoding="utf-8") as f:
        f.write(result.get("text", ""))

    print(f"Saved Whisper results to {output_dir} ({len(segments)} segments, {len(words)} words)")


def run_whisper_transcription(
    audio_file: str, model_name: str = "base", output_dir: Optional[str] = None
) -> str:
    """
    Transcribe audio with Whisper (word timestamps on) and save outputs.
    
    Returns:
        Path to the output directory containing whisper_segments.json, whisper_words.json, whisper_transcription.txt
    """
    if output_dir is None:
        base = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(os.path.dirname(audio_file), f"{base}_whisper_results")

    print(f"Loading audio with librosa: {audio_file}")
    audio, _ = librosa.load(audio_file, sr=16000, mono=True, dtype="float32")

    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    print("Transcribing with word timestamps...")
    result = model.transcribe(
        audio,
        language="vi",
        word_timestamps=True,
        task="transcribe",
        verbose=False,
    )

    save_whisper_results(result, output_dir)
    return output_dir


def cut_audio_segments(
    audio_file: str,
    sentence_timestamps: List[Tuple[float, float, str]],
    output_dir: str,
    padding: float = 0.3,
    add_padding: Optional[float] = None,  # backward compatibility for older callers
) -> None:
    """Cut audio into per-sentence WAV/text files."""
    if add_padding is not None:
        padding = add_padding
    
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = librosa.load(audio_file, sr=None, mono=True)

    for i, (start, end, text) in enumerate(sentence_timestamps, 1):
        start_s = max(0, int((start - padding) * sr))
        end_s = min(len(audio), int((end + padding) * sr))
        segment = audio[start_s:end_s]

        wav_path = os.path.join(output_dir, f"sentence_{i:05d}.wav")
        sf.write(wav_path, segment, sr)

        txt_path = os.path.join(output_dir, f"sentence_{i:05d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Cut {len(sentence_timestamps)} segments into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Whisper helpers")
    parser.add_argument("cmd", choices=["transcribe"], help="Supported command")
    parser.add_argument("audio_file")
    parser.add_argument("--model", default="base")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.cmd == "transcribe":
        run_whisper_transcription(args.audio_file, args.model, args.output_dir)


if __name__ == "__main__":
    main()

