#!/usr/bin/env python3
"""
End-to-end word-level pipeline:
1) Whisper transcription -> saves to whisper_output_xaxoi/<base>/whisper_words.json
2) Word-level matching (match_sentence_words.py)
3) Audio cutting from matches (cut_audio_from_word_matches.py)
"""

import argparse
from pathlib import Path

from align_vietnamese_audio import run_whisper_transcription, cut_audio_segments
from match_sentence_words import match_sentences_using_words, save_results as save_word_matches
from cut_audio_from_word_matches import process_matches_for_cutting, load_word_matches


def run_pipeline(
    audio_file: Path,
    text_file: Path,
    whisper_root: Path,
    output_root: Path,
    model: str = "base",
    padding: float = 0.3,
) -> None:
    base_name = audio_file.stem

    # Step 1: Whisper transcription
    whisper_dir = whisper_root / base_name
    print(f"[1/3] Whisper → {whisper_dir}")
    run_whisper_transcription(str(audio_file), model_name=model, output_dir=str(whisper_dir))
    words_file = whisper_dir / "whisper_words.json"
    if not words_file.exists():
        raise FileNotFoundError(f"whisper_words.json not found at {words_file}")

    # Step 2: Word-level matching
    output_root.mkdir(parents=True, exist_ok=True)
    matches_path = output_root / f"{base_name}_word_level_matches.json"
    print(f"[2/3] Matching sentences → {matches_path}")
    results = match_sentences_using_words(
        text_file_path=str(text_file),
        words_file_path=str(words_file),
        min_similarity=0.6,
        max_word_gap=5,
        max_search_window=500,
        strict_sequential=True,
        max_word_jump=100,
        max_time_jump=30.0,
    )
    save_word_matches(results, str(matches_path))

    # Step 3: Cut audio from matches
    audio_segments_dir = output_root / "audio_segments_method_w" / base_name
    print(f"[3/3] Cutting audio → {audio_segments_dir}")
    matches = load_word_matches(str(matches_path))
    sentence_timestamps, _stats = process_matches_for_cutting(matches)
    if not sentence_timestamps:
        raise RuntimeError("No valid sentence timestamps produced from matches; nothing to cut.")
    audio_segments_dir.mkdir(parents=True, exist_ok=True)
    cut_audio_segments(str(audio_file), sentence_timestamps, str(audio_segments_dir), add_padding=padding)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Word-level pipeline: whisper -> match -> cut")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("text_file", type=Path, help="Text file with one sentence per line")
    parser.add_argument("--whisper-root", type=Path, default=Path("whisper_output_xaxoi"))
    parser.add_argument("--output-root", type=Path, default=Path("output_xaxoi"))
    parser.add_argument("--model", default="base")
    parser.add_argument("--padding", type=float, default=0.3)
    args = parser.parse_args()

    run_pipeline(
        audio_file=args.audio_file,
        text_file=args.text_file,
        whisper_root=args.whisper_root,
        output_root=args.output_root,
        model=args.model,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()

