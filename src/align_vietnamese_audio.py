#!/usr/bin/env python3
"""
Minimal Whisper-based Vietnamese audio alignment and cutting.

Core flow:
1) Transcribe audio with Whisper (word-level timestamps saved).
2) Align provided sentences to word timestamps (or use Whisper-detected sentences).
3) Cut audio into per-sentence WAV files.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
import whisper


def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_whisper_results(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    segments = []
    words = []
    for seg in result.get("segments", []):
        segments.append(
            {"id": seg.get("id"), "start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", "").strip()}
        )
        for word in seg.get("words", []):
            words.append(
                {"word": word.get("word", "").strip(), "start": word.get("start"), "end": word.get("end")}
            )

    with open(os.path.join(output_dir, "whisper_segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "whisper_words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "whisper_transcription.txt"), "w", encoding="utf-8") as f:
        f.write(result.get("text", ""))

    print(f"Saved Whisper results to {output_dir} ({len(segments)} segments, {len(words)} words)")


def load_whisper_results(output_dir: str) -> Tuple[List[Tuple[float, float, str]], List[dict], str]:
    with open(os.path.join(output_dir, "whisper_segments.json"), "r", encoding="utf-8") as f:
        segments_json = json.load(f)
    with open(os.path.join(output_dir, "whisper_words.json"), "r", encoding="utf-8") as f:
        words_json = json.load(f)
    with open(os.path.join(output_dir, "whisper_transcription.txt"), "r", encoding="utf-8") as f:
        transcription = f.read()

    segments = [(seg.get("start", 0.0), seg.get("end", 0.0), seg.get("text", "").strip()) for seg in segments_json]
    return segments, words_json, transcription


def run_whisper_transcription(audio_file: str, model_name: str = "base", output_dir: Optional[str] = None):
    if output_dir is None:
        base = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(os.path.dirname(audio_file), f"{base}_whisper_results")

    print(f"Loading audio with librosa: {audio_file}")
    audio, _ = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)

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

    segments = [(seg["start"], seg["end"], seg["text"].strip()) for seg in result.get("segments", []) if seg.get("text")]
    words = [
        {"word": w.get("word", "").strip(), "start": w.get("start"), "end": w.get("end")}
        for seg in result.get("segments", [])
        for w in seg.get("words", [])
    ]
    return segments, words, result.get("text", ""), output_dir


def find_sentence_in_words(sentence: str, words: List[dict], start_idx: int = 0) -> Optional[Tuple[int, int]]:
    norm_sentence = normalize_text(sentence)
    tokens = norm_sentence.split()
    if not tokens:
        return None

    best = None
    best_score = 0.0

    for i in range(start_idx, min(len(words), start_idx + 200)):
        end = min(len(words), i + len(tokens) + 10)
        audio_tokens = [normalize_text(w["word"]) for w in words[i:end]]
        joined_audio = " ".join(audio_tokens)
        score = SequenceMatcher(None, " ".join(tokens), joined_audio).ratio()
        if score > best_score:
            best_score = score
            best = (i, i + len(tokens))
        if best_score > 0.95:
            break

    return best if best_score >= 0.4 else None


def align_sentences_to_words(sentences: List[str], words: List[dict]) -> List[Tuple[float, float, str]]:
    timestamps = []
    word_idx = 0

    for sentence in sentences:
        match = find_sentence_in_words(sentence, words, word_idx)
        if match:
            start_i, end_i = match
            start_t = words[start_i]["start"]
            end_t = words[end_i - 1]["end"]
            timestamps.append((start_t, end_t, sentence))
            word_idx = end_i
        else:
            # fallback estimate
            start_t = words[word_idx]["start"] if word_idx < len(words) else (timestamps[-1][1] if timestamps else 0.0)
            dur = max(0.5, len(sentence.split()) / 3.5)
            timestamps.append((start_t, start_t + dur, sentence))
            word_idx = min(len(words), word_idx + len(sentence.split()))
    return timestamps


def align_from_whisper_results(whisper_dir: str, sentences_file: Optional[str], use_detected: bool) -> Tuple[List[Tuple[float, float, str]], str]:
    segments, words, transcription = load_whisper_results(whisper_dir)

    if use_detected or not sentences_file:
        # Use Whisper segments as-is
        return [(s, e, txt) for s, e, txt in segments if txt.strip()], transcription

    with open(sentences_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    timestamps = align_sentences_to_words(sentences, words)
    return timestamps, transcription


def cut_audio_segments(
    audio_file: str,
    sentence_timestamps: List[Tuple[float, float, str]],
    output_dir: str,
    padding: float = 0.3,
    add_padding: Optional[float] = None,  # backward compatibility for older callers
) -> None:
    # accept legacy keyword add_padding if provided
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


def save_results(sentence_timestamps: List[Tuple[float, float, str]], transcription: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "timestamps.txt"), "w", encoding="utf-8") as f:
        for start, end, sentence in sentence_timestamps:
            f.write(f"{start:.3f}\t{end:.3f}\t{sentence}\n")
    with open(os.path.join(output_dir, "transcription.txt"), "w", encoding="utf-8") as f:
        f.write(transcription)


def main():
    parser = argparse.ArgumentParser(description="Whisper-based Vietnamese alignment and cutting")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_trans = sub.add_parser("transcribe", help="Run Whisper and save segments/words")
    p_trans.add_argument("audio_file")
    p_trans.add_argument("--model", default="base")
    p_trans.add_argument("--output-dir", default=None)

    p_align = sub.add_parser("align", help="Align from saved Whisper results")
    p_align.add_argument("whisper_dir")
    p_align.add_argument("--sentences-file", help="Text file with one sentence per line")
    p_align.add_argument("--use-detected", action="store_true", help="Use Whisper-detected segments instead of sentences file")
    p_align.add_argument("--output-dir", default="audio_segments")
    p_align.add_argument("--audio-file", help="Optional audio file to cut segments after alignment")

    p_full = sub.add_parser("full", help="Transcribe, align, and cut in one step")
    p_full.add_argument("audio_file")
    p_full.add_argument("sentences_file", nargs="?", help="Text file with one sentence per line")
    p_full.add_argument("--use-detected", action="store_true", help="Use Whisper-detected sentences")
    p_full.add_argument("--model", default="base")
    p_full.add_argument("--output-dir", default="audio_segments")

    args = parser.parse_args()

    if args.cmd == "transcribe":
        run_whisper_transcription(args.audio_file, args.model, args.output_dir)
        return

    if args.cmd == "align":
        timestamps, transcription = align_from_whisper_results(args.whisper_dir, args.sentences_file, args.use_detected)
        save_results(timestamps, transcription, args.output_dir)
        if args.audio_file:
            cut_audio_segments(args.audio_file, timestamps, args.output_dir)
        return

    if args.cmd == "full":
        segments, words, transcription, whisper_dir = run_whisper_transcription(args.audio_file, args.model)
        timestamps, _ = align_from_whisper_results(whisper_dir, args.sentences_file, args.use_detected)
        save_results(timestamps, transcription, args.output_dir)
        cut_audio_segments(args.audio_file, timestamps, args.output_dir)
        return


if __name__ == "__main__":
    main()
