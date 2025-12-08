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
            # No matching - use original flow (merge until periods, then split)
            sentence_timestamps = []
            current_merged_text = ""
            current_merged_start = None
            current_merged_end = None
            current_merged_words = []
            
            for seg_start, seg_end, text in segments_list:
                if not text:
                    continue
                
                # Get words for this segment
                segment_words = []
                seg_start_time = seg_start
                seg_end_time = seg_end
                
                # Find words that fall within this segment
                for word_info in words_list:
                    word_start = word_info.get("start", 0)
                    word_end = word_info.get("end", 0)
                    if word_start >= seg_start and word_end <= seg_end:
                        segment_words.append(word_info)
                
                # Merge with current accumulated text
                if current_merged_text:
                    current_merged_text += " " + text
                else:
                    current_merged_text = text
                    current_merged_start = seg_start
                
                current_merged_end = seg_end
                current_merged_words.extend(segment_words)
                
                # Check if this merged text ends with a period
                if current_merged_text.rstrip().endswith('.'):
                    # Split at periods and add all sentences
                    merged_segment = {
                        "text": current_merged_text,
                        "start": current_merged_start,
                        "end": current_merged_end
                    }
                    split_sentences = split_segment_at_periods(merged_segment, current_merged_words)
                    sentence_timestamps.extend(split_sentences)
                    
                    # Reset for next merge
                    current_merged_text = ""
                    current_merged_start = None
                    current_merged_end = None
                    current_merged_words = []
            
            # If there's remaining text without a period at the end, add it as one sentence
            if current_merged_text:
                sentence_timestamps.append((current_merged_start, current_merged_end, current_merged_text))
            
            print(f"Đã phát hiện {len(sentence_timestamps)} câu trong audio (chỉ cắt tại dấu chấm)")
            print(f"Detected {len(sentence_timestamps)} sentences in audio (only cut at periods)")
        
        return sentence_timestamps, full_transcription
    
    # Otherwise, use provided transcript and align
    if sentences_file is None:
        raise ValueError("sentences_file must be provided when use_detected_sentences=False")
    
    # Read sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Đã nhận diện {len(words_list)} từ trong audio")
    print(f"Recognized {len(words_list)} words in audio")
    print(f"Đang căn chỉnh {len(sentences)} câu với audio...")
    print(f"Aligning {len(sentences)} sentences with audio...")
    
    # Align sentences
    sentence_timestamps = []
    word_idx = 0
    failed_alignments = []
    
    for i, sentence in enumerate(sentences):
        if word_idx >= len(words_list):
            # Estimate remaining sentences
            if sentence_timestamps:
                last_end = sentence_timestamps[-1][1]
                estimated_duration = len(sentence.split()) / 3.5  # ~3.5 words/sec
                sentence_timestamps.append((last_end, last_end + estimated_duration, sentence))
            continue
        
        # Find sentence in transcription
        start_idx, end_idx, confidence = find_sentence_in_transcription(
            sentence, words_list, word_idx
        )
        
        if start_idx is not None and end_idx is not None:
            start_time = words_list[start_idx]['start']
            end_time = words_list[end_idx - 1]['end'] if end_idx > 0 else words_list[start_idx]['end']
            sentence_timestamps.append((start_time, end_time, sentence))
            word_idx = end_idx
            
            if (i + 1) % 100 == 0:
                print(f"  Đã căn chỉnh {i+1}/{len(sentences)} câu (độ tin cậy: {confidence:.2%})")
                print(f"  Aligned {i+1}/{len(sentences)} sentences (confidence: {confidence:.2%})")
        else:
            # Fallback: estimate based on previous alignment
            if sentence_timestamps:
                last_end = sentence_timestamps[-1][1]
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((last_end, last_end + estimated_duration, sentence))
                failed_alignments.append(i)
            else:
                # First sentence - use first word timestamp
                start_time = words_list[word_idx]['start'] if words_list else 0
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((start_time, start_time + estimated_duration, sentence))
                failed_alignments.append(i)
            
            word_idx += len(sentence.split())  # Skip some words
    
    if failed_alignments:
        print(f"\nCảnh báo: {len(failed_alignments)} câu không thể căn chỉnh chính xác (đã ước tính)")
        print(f"Warning: {len(failed_alignments)} sentences could not be aligned (estimated)")
    
    return sentence_timestamps, full_transcription


def align_vietnamese_audio_improved(audio_file, sentences_file=None, model_name="base", use_detected_sentences=False, match_with_real_sentences=False):
    """
    Improved alignment for Vietnamese audio and text.
    This function runs Whisper and then performs alignment.
    For better performance, use run_whisper_transcription() and align_from_whisper_results() separately.
    
    Args:
        audio_file: Path to audio file
        sentences_file: Path to file with Vietnamese sentences (one per line). Optional if use_detected_sentences=True.
        model_name: Whisper model to use
        use_detected_sentences: If True, use sentences detected by Whisper instead of provided transcript
        match_with_real_sentences: If True, match Whisper segments with real sentences from file
    
    Returns:
        Tuple of (sentence_timestamps, full_transcription)
    """
    # Run Whisper transcription and save results
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    whisper_output_dir = os.path.join(os.path.dirname(audio_file), f"{base_name}_whisper_results")
    
    _, _, _, _ = run_whisper_transcription(audio_file, model_name, whisper_output_dir)
    
    # Perform alignment from saved results
    return align_from_whisper_results(whisper_output_dir, sentences_file, use_detected_sentences, match_with_real_sentences)


def cut_audio_segments(audio_file, sentence_timestamps, output_dir, add_padding=0.3):
    """
    Cut audio into segments with padding.
    Uses librosa to avoid ffmpeg issues.
    
    Args:
        audio_file: Path to input audio file
        sentence_timestamps: List of (start, end, sentence) tuples
        output_dir: Directory to save segments
        add_padding: Seconds to add before/after each segment
    """
    print(f"Đang tải audio: {audio_file}...")
    print(f"Loading audio: {audio_file}...")
    sys.stdout.flush()
    
    # Try librosa first (works without ffmpeg)
    audio_array = None
    sample_rate = None
    
    if HAS_LIBROSA:
        try:
            audio_array, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            print(f"✓ Đã tải audio bằng librosa: {len(audio_array)} samples @ {sample_rate}Hz")
            print(f"✓ Successfully loaded audio with librosa: {len(audio_array)} samples @ {sample_rate}Hz")
        except Exception as e:
            if HAS_PYDUB:
                print(f"⚠ Librosa failed: {e}, trying pydub...")
                print(f"⚠ Librosa failed: {e}, trying pydub...")
            else:
                print(f"⚠ Librosa failed: {e}")
                print(f"⚠ Librosa failed: {e}")
            audio_array = None
    
    # Fallback to pydub if librosa failed
    if audio_array is None:
        if HAS_PYDUB:
            try:
                audio_seg = AudioSegment.from_file(audio_file)
                # Convert pydub AudioSegment to numpy array for consistent processing
                audio_array = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
                if audio_seg.channels == 2:
                    # Convert stereo to mono
                    audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
                sample_rate = audio_seg.frame_rate
                # Normalize to [-1, 1] range
                if audio_array.max() > 1.0:
                    audio_array = audio_array / (2 ** (audio_seg.sample_width * 8 - 1))
            except Exception as e:
                print(f"❌ Không thể tải audio: {e}")
                print(f"❌ Cannot load audio: {e}")
                raise RuntimeError(f"Failed to load audio file. Both librosa and pydub failed. Error: {e}")
        else:
            raise RuntimeError(f"Failed to load audio file. librosa failed and pydub is not available.")
    
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
