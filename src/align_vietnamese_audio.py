#!/usr/bin/env python3
"""
Improved Vietnamese audio-text alignment using Whisper with better matching strategies.
Aligns Vietnamese audio with Vietnamese text files and cuts audio into sentence segments.
"""

import os
import sys
import re
import json
from difflib import SequenceMatcher
import unicodedata
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available, will try other methods")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("Warning: pydub not available, will use librosa only")


def remove_diacritics(text):
    """Remove Vietnamese diacritics for better matching."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def normalize_text(text):
    """Normalize Vietnamese text for matching."""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation for matching
    text = re.sub(r'[.,!?:;]', '', text)
    return text.strip().lower()


def match_whisper_to_real_sentences(whisper_segments, real_sentences, word_segments):
    """
    Match Whisper-detected segments to real sentences from text file using fuzzy matching.
    Uses sliding window approach for better matching. Handles ~60% letter accuracy.
    Matches whole real sentences to one or more Whisper segments.
    
    Args:
        whisper_segments: List of (start, end, text) tuples from Whisper
        real_sentences: List of real sentences from text file
        word_segments: List of word dicts with 'word', 'start', 'end' from Whisper
    
    Returns:
        List of (start, end, real_sentence_text) tuples - one per real sentence
    """
    matched_sentences = []
    real_sent_idx = 0
    whisper_seg_idx = 0
    
    # Build a normalized version of real sentences for matching
    normalized_real = []
    for sent in real_sentences:
        normalized_real.append(normalize_text(sent))
    
    # Sliding window size for matching (look ahead/behind)
    window_size = 15
    
    while real_sent_idx < len(real_sentences):
        real_sentence = real_sentences[real_sent_idx]
        normalized_real_sent = normalized_real[real_sent_idx]
        
        # Try to match this real sentence with one or more Whisper segments
        best_match_start_idx = None
        best_match_end_idx = None
        best_score = 0
        accumulated_whisper_text = ""
        
        # Search in window around current whisper position
        search_start = max(0, whisper_seg_idx)
        search_end = min(len(whisper_segments), whisper_seg_idx + window_size)
        
        # Try matching with single segment or multiple consecutive segments
        for start_idx in range(search_start, search_end):
            accumulated_text = ""
            accumulated_start = whisper_segments[start_idx][0] if start_idx < len(whisper_segments) else None
            accumulated_end = None
            
            # Try matching with 1 to 5 consecutive segments
            for num_segments in range(1, min(6, len(whisper_segments) - start_idx + 1)):
                if start_idx + num_segments - 1 >= len(whisper_segments):
                    break
                
                # Accumulate text from consecutive segments
                segment_texts = []
                for i in range(num_segments):
                    seg_idx = start_idx + i
                    if seg_idx < len(whisper_segments):
                        seg_start, seg_end, seg_text = whisper_segments[seg_idx]
                        segment_texts.append(seg_text)
                        if accumulated_start is None:
                            accumulated_start = seg_start
                        accumulated_end = seg_end
                
                accumulated_text = " ".join(segment_texts)
                normalized_whisper = normalize_text(accumulated_text)
                
                # Strategy 1: Check if normalized texts are similar enough
                similarity = SequenceMatcher(None, normalized_whisper, normalized_real_sent).ratio()
                
                # Strategy 2: Check word overlap (accounting for ~60% accuracy)
                whisper_words = set(normalized_whisper.split())
                real_words = set(normalized_real_sent.split())
                if len(whisper_words) > 0 and len(real_words) > 0:
                    word_overlap = len(whisper_words & real_words) / max(len(whisper_words), len(real_words))
                else:
                    word_overlap = 0
                
                # Strategy 3: Check without diacritics
                whisper_no_diac = remove_diacritics(normalized_whisper)
                real_no_diac = remove_diacritics(normalized_real_sent)
                no_diac_similarity = SequenceMatcher(None, whisper_no_diac, real_no_diac).ratio()
                
                # Strategy 4: Check substring match (for partial matches)
                substring_match = 0
                if len(normalized_whisper) > 5 and len(normalized_real_sent) > 5:
                    if normalized_whisper in normalized_real_sent or normalized_real_sent in normalized_whisper:
                        substring_match = 0.5
                    # Check if significant portion matches
                    min_len = min(len(normalized_whisper), len(normalized_real_sent))
                    if min_len > 0:
                        common_chars = sum(1 for a, b in zip(normalized_whisper[:min_len], normalized_real_sent[:min_len]) if a == b)
                        substring_match = max(substring_match, common_chars / min_len * 0.3)
                
                # Combined score (weighted) - looser threshold
                combined_score = (similarity * 0.3 + word_overlap * 0.3 + no_diac_similarity * 0.2 + substring_match * 0.2)
                
                # Prefer matches that are closer to expected length
                length_ratio = min(len(normalized_whisper), len(normalized_real_sent)) / max(len(normalized_whisper), len(normalized_real_sent), 1)
                combined_score *= (0.7 + 0.3 * length_ratio)  # Boost if lengths are similar
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_start_idx = start_idx
                    best_match_end_idx = start_idx + num_segments - 1
                    accumulated_whisper_text = accumulated_text
        
        # Match if score is above threshold (much lower threshold - 0.15)
        if best_score > 0.15 and best_match_start_idx is not None:
            # Get timestamps from matched segments
            match_start = whisper_segments[best_match_start_idx][0]
            match_end = whisper_segments[best_match_end_idx][1]
            
            # Use real sentence text but Whisper timestamps
            matched_sentences.append((match_start, match_end, real_sentence))
            whisper_seg_idx = best_match_end_idx + 1  # Move past matched segments
            real_sent_idx += 1
            
            if len(matched_sentences) % 50 == 0:
                print(f"  Đã khớp {len(matched_sentences)}/{len(real_sentences)} câu (độ tương đồng: {best_score:.2%})")
                print(f"  Matched {len(matched_sentences)}/{len(real_sentences)} sentences (similarity: {best_score:.2%})")
        else:
            # No good match found - estimate timestamp and use real sentence
            if matched_sentences:
                # Estimate based on previous matches
                last_end = matched_sentences[-1][1]
                # Estimate ~3.5 words per second
                estimated_duration = len(real_sentence.split()) / 3.5
                estimated_start = last_end
                estimated_end = last_end + estimated_duration
                matched_sentences.append((estimated_start, estimated_end, real_sentence))
            elif whisper_seg_idx < len(whisper_segments):
                # Use first available Whisper segment timestamp
                match_start = whisper_segments[whisper_seg_idx][0]
                estimated_duration = len(real_sentence.split()) / 3.5
                matched_sentences.append((match_start, match_start + estimated_duration, real_sentence))
            else:
                # Fallback: use zero timestamp
                estimated_duration = len(real_sentence.split()) / 3.5
                matched_sentences.append((0, estimated_duration, real_sentence))
            
            real_sent_idx += 1
            # Don't advance whisper_seg_idx too much if we're way ahead
            if whisper_seg_idx < len(whisper_segments) and whisper_seg_idx < real_sent_idx + 10:
                pass  # Keep current position
            else:
                whisper_seg_idx = max(0, real_sent_idx - 5)  # Reset to reasonable position
    
    return matched_sentences


def split_segment_at_periods(segment, words_in_segment):
    """
    Split a segment into sentences at period boundaries.
    
    Args:
        segment: Segment dict with 'text', 'start', 'end'
        words_in_segment: List of word dicts with 'word', 'start', 'end' for this segment
    
    Returns:
        List of (start, end, sentence_text) tuples
    """
    text = segment.get("text", "").strip()
    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)
    
    if not text:
        return []
    
    # Split text at periods, keeping the period with the sentence
    # Split on period followed by space or end of string
    sentences = re.split(r'\.(?=\s|$)', text)
    
    # Add period back to each sentence (except the last one if it doesn't end with period)
    sentence_list = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period if it's not the last sentence, or if the original text ended with period
        if i < len(sentences) - 1:
            sentence += '.'
        elif text.rstrip().endswith('.'):
            sentence += '.'
        
        sentence_list.append(sentence)
    
    # If no periods found, return original segment
    if len(sentence_list) <= 1:
        return [(segment_start, segment_end, text)]
    
    # Estimate timestamps for each sentence using word positions
    result_segments = []
    word_idx = 0
    text_processed = 0
    
    for sentence in sentence_list:
        if word_idx >= len(words_in_segment):
            # Estimate remaining sentences
            if result_segments:
                last_end = result_segments[-1][1]
                estimated_duration = len(sentence.split()) / 3.5
                result_segments.append((last_end, last_end + estimated_duration, sentence))
            else:
                result_segments.append((segment_start, segment_end, sentence))
            continue
        
        # Find where this sentence starts in the word list
        sentence_words = sentence.split()
        sentence_start_idx = word_idx
        
        # Try to match sentence words with segment words
        matched_words = 0
        sentence_end_idx = word_idx
        
        for sent_word in sentence_words:
            if sentence_end_idx >= len(words_in_segment):
                break
            
            # Simple matching - check if word appears in segment word
            seg_word = words_in_segment[sentence_end_idx].get("word", "").strip()
            if sent_word.lower() in seg_word.lower() or seg_word.lower() in sent_word.lower():
                matched_words += 1
                sentence_end_idx += 1
            else:
                # Try next word
                sentence_end_idx += 1
        
        # Calculate timestamps
        if sentence_end_idx > sentence_start_idx:
            sent_start_time = words_in_segment[sentence_start_idx].get("start", segment_start)
            sent_end_time = words_in_segment[sentence_end_idx - 1].get("end", segment_end)
        else:
            # Fallback: estimate based on position
            if result_segments:
                sent_start_time = result_segments[-1][1]
            else:
                sent_start_time = segment_start
            estimated_duration = len(sentence_words) / 3.5
            sent_end_time = sent_start_time + estimated_duration
        
        result_segments.append((sent_start_time, sent_end_time, sentence))
        word_idx = sentence_end_idx
    
    return result_segments


def save_whisper_results(whisper_result, output_dir):
    """
    Save Whisper transcription results to JSON files for later use.
    This saves the heavy work (Whisper transcription) so it can be reused.
    
    Args:
        whisper_result: Result dict from Whisper model.transcribe()
        output_dir: Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save segments
    segments_file = os.path.join(output_dir, "whisper_segments.json")
    segments_data = []
    for segment in whisper_result.get("segments", []):
        segments_data.append({
            "id": segment.get("id"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text", "").strip()
        })
    
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)
    
    # Save word timestamps
    words_file = os.path.join(output_dir, "whisper_words.json")
    words_data = []
    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            words_data.append({
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start"),
                "end": word_info.get("end")
            })
    
    with open(words_file, 'w', encoding='utf-8') as f:
        json.dump(words_data, f, ensure_ascii=False, indent=2)
    
    # Save full transcription
    transcription_file = os.path.join(output_dir, "whisper_transcription.txt")
    with open(transcription_file, 'w', encoding='utf-8') as f:
        f.write(whisper_result.get("text", ""))
    
    print(f"✓ Đã lưu kết quả Whisper vào: {output_dir}")
    print(f"✓ Saved Whisper results to: {output_dir}")
    print(f"  - Segments: {len(segments_data)}")
    print(f"  - Words: {len(words_data)}")
    print(f"  - Full transcription saved")


def load_whisper_results(output_dir):
    """
    Load Whisper transcription results from saved JSON files.
    
    Args:
        output_dir: Directory containing saved Whisper results
    
    Returns:
        Tuple of (segments_list, words_list, full_transcription)
        segments_list: List of (start, end, text) tuples
        words_list: List of dicts with 'word', 'start', 'end'
        full_transcription: Full transcription text string
    """
    segments_file = os.path.join(output_dir, "whisper_segments.json")
    words_file = os.path.join(output_dir, "whisper_words.json")
    transcription_file = os.path.join(output_dir, "whisper_transcription.txt")
    
    if not os.path.exists(segments_file):
        raise FileNotFoundError(f"Whisper segments file not found: {segments_file}")
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Whisper words file not found: {words_file}")
    if not os.path.exists(transcription_file):
        raise FileNotFoundError(f"Whisper transcription file not found: {transcription_file}")
    
    # Load segments
    with open(segments_file, 'r', encoding='utf-8') as f:
        segments_data = json.load(f)
    
    segments_list = []
    for seg in segments_data:
        segments_list.append((
            seg.get("start", 0),
            seg.get("end", 0),
            seg.get("text", "").strip()
        ))
    
    # Load words
    with open(words_file, 'r', encoding='utf-8') as f:
        words_data = json.load(f)
    
    words_list = []
    for word_info in words_data:
        words_list.append({
            "word": word_info.get("word", "").strip(),
            "start": word_info.get("start", 0),
            "end": word_info.get("end", 0)
        })
    
    # Load full transcription
    with open(transcription_file, 'r', encoding='utf-8') as f:
        full_transcription = f.read()
    
    print(f"✓ Đã tải kết quả Whisper từ: {output_dir}")
    print(f"✓ Loaded Whisper results from: {output_dir}")
    print(f"  - Segments: {len(segments_list)}")
    print(f"  - Words: {len(words_list)}")
    
    return segments_list, words_list, full_transcription


def find_sentence_in_transcription(sentence, transcription_words, start_idx=0):
    """
    Find sentence in transcription using multiple strategies.
    Returns (start_word_idx, end_word_idx, confidence)
    """
    sentence_normalized = normalize_text(sentence)
    sentence_words = sentence_normalized.split()
    
    if not sentence_words:
        return None, None, 0
    
    best_match = None
    best_score = 0
    best_start = start_idx
    best_end = start_idx
    
    # Strategy 1: Exact word sequence matching
    for i in range(start_idx, min(start_idx + 200, len(transcription_words) - len(sentence_words) + 1)):
        matched = 0
        trans_idx = i
        
        for sent_word in sentence_words[:15]:  # Check first 15 words
            if trans_idx >= len(transcription_words):
                break
            
            trans_word = normalize_text(transcription_words[trans_idx]['word'])
            
            # Exact match
            if sent_word == trans_word:
                matched += 1
                trans_idx += 1
            # Partial match (one word contains the other)
            elif sent_word in trans_word or trans_word in sent_word:
                matched += 0.8
                trans_idx += 1
            # Similarity match
            elif len(sent_word) > 2 and len(trans_word) > 2:
                similarity = SequenceMatcher(None, sent_word, trans_word).ratio()
                if similarity > 0.75:
                    matched += similarity
                    trans_idx += 1
                else:
                    # Try without diacritics
                    sent_no_diac = remove_diacritics(sent_word)
                    trans_no_diac = remove_diacritics(trans_word)
                    if sent_no_diac == trans_no_diac:
                        matched += 0.9
                        trans_idx += 1
                    else:
                        break
            else:
                # Allow skipping 1-2 words for minor mismatches
                if trans_idx + 1 < len(transcription_words):
                    next_word = normalize_text(transcription_words[trans_idx + 1]['word'])
                    if sent_word == next_word or sent_word in next_word:
                        trans_idx += 2
                        matched += 0.7
                    else:
                        break
                else:
                    break
        
        score = matched / len(sentence_words[:15])
        if score > best_score:
            best_score = score
            best_start = i
            best_end = trans_idx
    
    # Only return if confidence is reasonable
    if best_score > 0.4:  # At least 40% match
        return best_start, best_end, best_score
    
    return None, None, 0


def run_whisper_transcription(audio_file, model_name="base", output_dir=None):
    """
    Run Whisper transcription and save results to files.
    This is the heavy work that should be done once.
    
    Args:
        audio_file: Path to audio file
        model_name: Whisper model to use
        output_dir: Directory to save Whisper results (if None, uses same dir as audio)
    
    Returns:
        Tuple of (segments_list, words_list, full_transcription)
    """
    if output_dir is None:
        # Use same directory as audio file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(os.path.dirname(audio_file), f"{base_name}_whisper_results")
    
    # Use librosa to load audio directly (bypasses ffmpeg)
    audio_array = None
    audio_file_for_whisper = audio_file
    conversion_success = False
    
    if HAS_LIBROSA:
        try:
            print("Đang tải audio bằng librosa (bỏ qua ffmpeg)...")
            print("Loading audio with librosa (bypass ffmpeg)...")
            sys.stdout.flush()
            
            # Load audio with librosa (16kHz mono, as Whisper expects)
            audio_array, sr = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)
            
            # Verify the audio array format
            if len(audio_array) == 0:
                raise Exception("Loaded audio array is empty")
            
            conversion_success = True
            print(f"✓ Đã tải audio bằng librosa: {len(audio_array)} samples @ {sr}Hz")
            print(f"✓ Successfully loaded audio with librosa: {len(audio_array)} samples @ {sr}Hz")
        except Exception as e:
            print(f"⚠ Librosa thất bại: {type(e).__name__}: {e}")
            print(f"⚠ Librosa failed: {type(e).__name__}: {e}")
            sys.stdout.flush()
    
    # If librosa failed, warn but continue with original file
    if not conversion_success:
        print("⚠ Cảnh báo: Không thể tải audio bằng librosa, thử dùng file gốc...")
        print("⚠ Warning: Could not load audio with librosa, trying original file...")
        sys.stdout.flush()
        audio_file_for_whisper = audio_file
    
    print(f"Đang tải mô hình Whisper: {model_name}...")
    print(f"Loading Whisper model: {model_name}...")
    sys.stdout.flush()
    model = whisper.load_model(model_name)
    
    print("Đang chuyển đổi giọng nói sang văn bản (có thể mất vài phút)...")
    print("Transcribing audio (this may take a few minutes)...")
    sys.stdout.flush()
    
    try:
        # If we have audio_array from librosa, use it directly
        if audio_array is not None:
            result = model.transcribe(
                audio_array,
                language="vi",
                word_timestamps=True,
                task="transcribe",
                verbose=False
            )
        else:
            # Otherwise use file path (may fail if ffmpeg broken)
            result = model.transcribe(
                audio_file_for_whisper,
                language="vi",
                word_timestamps=True,
                task="transcribe",
                verbose=False
            )
    except Exception as e:
        print(f"\n❌ Lỗi khi chuyển đổi giọng nói: {type(e).__name__}: {e}")
        print(f"❌ Error during transcription: {type(e).__name__}: {e}")
        sys.stdout.flush()
        raise RuntimeError(f"Whisper transcription failed. Error: {e}") from e
    
    # Save results
    save_whisper_results(result, output_dir)
    
    # Extract data in the format expected by alignment functions
    segments_list = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if text:
            segments_list.append((
                segment.get("start", 0),
                segment.get("end", 0),
                text
            ))
    
    words_list = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words_list.append({
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0)
            })
    
    full_transcription = result.get("text", "")
    
    return segments_list, words_list, full_transcription, output_dir


def align_from_whisper_results(whisper_dir, sentences_file=None, use_detected_sentences=False, match_with_real_sentences=False):
    """
    Perform alignment and splitting using saved Whisper results.
    This is the lighter weight work that can be done multiple times with different parameters.
    
    Args:
        whisper_dir: Directory containing saved Whisper results
        sentences_file: Path to file with Vietnamese sentences (one per line). Optional if use_detected_sentences=True.
        use_detected_sentences: If True, use sentences detected by Whisper instead of provided transcript
        match_with_real_sentences: If True, match Whisper segments with real sentences from file
    
    Returns:
        Tuple of (sentence_timestamps, full_transcription)
    """
    # Load Whisper results
    segments_list, words_list, full_transcription = load_whisper_results(whisper_dir)
    
    # If using detected sentences, match with real sentences first, then cut audio
    if use_detected_sentences:
        print("Đang sử dụng các câu được phát hiện bởi Whisper...")
        print("Using sentences detected by Whisper...")
        sys.stdout.flush()
        
        # If match_with_real_sentences is True and sentences_file is provided, match with real sentences FIRST
        if match_with_real_sentences and sentences_file:
            print("\nĐang khớp các câu thật với transcript Whisper...")
            print("Matching real sentences with Whisper transcript...")
            sys.stdout.flush()
            
            # Read real sentences
            with open(sentences_file, 'r', encoding='utf-8') as f:
                real_sentences = [line.strip() for line in f if line.strip()]
            
            # Match each whole real sentence to Whisper segments (looser matching)
            sentence_timestamps = match_whisper_to_real_sentences(
                segments_list, real_sentences, words_list
            )
            
            print(f"\nĐã khớp {len(sentence_timestamps)}/{len(real_sentences)} câu với văn bản thật")
            print(f"Matched {len(sentence_timestamps)}/{len(real_sentences)} sentences with real text")
            print("Đang chuẩn bị cắt audio dựa trên các câu đã khớp...")
            print("Preparing to cut audio based on matched sentences...")
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
    
    print(f"Đang cắt {len(sentence_timestamps)} đoạn audio...")
    print(f"Cutting {len(sentence_timestamps)} audio segments...")
    sys.stdout.flush()
    
    # Import soundfile for saving WAV files
    try:
        import soundfile as sf
        has_soundfile = True
    except ImportError:
        has_soundfile = False
        print("⚠ soundfile not available, will try alternative method...")
    
    for i, (start, end, sentence) in enumerate(sentence_timestamps):
        # Calculate sample indices with padding
        start_sample = max(0, int((start - add_padding) * sample_rate))
        end_sample = min(len(audio_array), int((end + add_padding) * sample_rate))
        
        # Extract segment
        segment = audio_array[start_sample:end_sample]
        
        output_file = os.path.join(output_dir, f"sentence_{i+1:05d}.wav")
        
        # Save segment using soundfile (preferred) or scipy
        try:
            if has_soundfile:
                sf.write(output_file, segment, sample_rate)
            else:
                # Fallback: use scipy.io.wavfile
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for WAV format
                    segment_int16 = (segment * 32767).astype(np.int16)
                    wavfile.write(output_file, int(sample_rate), segment_int16)
                except ImportError:
                    # Last resort: try pydub (may fail if ffmpeg broken)
                    if HAS_PYDUB:
                        try:
                            temp_seg = AudioSegment(
                                segment.tobytes(),
                                frame_rate=int(sample_rate),
                                channels=1,
                                sample_width=2
                            )
                            temp_seg.export(output_file, format="wav")
                        except Exception as e2:
                            raise RuntimeError(f"Cannot save audio segment: soundfile, scipy, and pydub all failed. Error: {e2}")
                    else:
                        raise RuntimeError(f"Cannot save audio segment: soundfile and scipy not available, and pydub is not installed.")
        except Exception as e:
            print(f"⚠ Lỗi khi lưu segment {i+1}: {e}")
            print(f"⚠ Error saving segment {i+1}: {e}")
            continue
        
        # Save text
        text_file = os.path.join(output_dir, f"sentence_{i+1:05d}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(sentence)
        
        if (i + 1) % 100 == 0:
            print(f"  Đã xử lý {i+1}/{len(sentence_timestamps)} câu...")
            print(f"  Processed {i+1}/{len(sentence_timestamps)} sentences...")
            sys.stdout.flush()
    
    print(f"\nHoàn thành! Đã cắt {len(sentence_timestamps)} đoạn audio.")
    print(f"Complete! Cut {len(sentence_timestamps)} audio segments.")


# ============================================================================
# METHOD 2: Replace wrong whisper transcriptions with correct text
# ============================================================================

def normalize_text_for_matching_method2(text):
    """Normalize text for fuzzy matching by removing diacritics and punctuation."""
    # Remove punctuation and extra spaces
    # In character class, put brackets at start/end to avoid escaping
    text = re.sub(r'[[\].,!?:;"\'(){}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def similarity_score_method2(text1, text2):
    """Calculate similarity score between two texts."""
    norm1 = normalize_text_for_matching_method2(text1)
    norm2 = normalize_text_for_matching_method2(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def load_text_file_method2(file_path):
    """Load text file and return list of sentences/lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by newlines first, then by sentence-ending punctuation if needed
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Further split long lines by sentence punctuation
    sentences = []
    for line in lines:
        # Split on sentence-ending punctuation but keep it
        parts = re.split(r'([.!?]+)', line)
        current = ""
        for i, part in enumerate(parts):
            current += part
            if part and part[0] in '.!?':
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
    
    return sentences


def align_segments_to_text_method2(whisper_segments, reference_sentences):
    """
    Align whisper segments to reference text and replace incorrect transcriptions.
    Combines multiple segments that match one sentence and fixes timestamps.
    
    Args:
        whisper_segments: List of whisper segment dicts with 'id', 'start', 'end', 'text'
        reference_sentences: List of correct sentences from text file
    
    Returns:
        List of corrected segments (may have fewer segments than input if combined)
    """
    corrected_segments = []
    ref_idx = 0
    whisper_idx = 0
    
    # Build normalized reference for matching
    normalized_ref = [normalize_text_for_matching_method2(sent) for sent in reference_sentences]
    
    # Maximum look-ahead window (search up to 20 segments ahead)
    max_window = 20
    
    while whisper_idx < len(whisper_segments) and ref_idx < len(reference_sentences):
        # Get current reference sentence
        ref_sentence = reference_sentences[ref_idx]
        normalized_ref_sent = normalized_ref[ref_idx]
        
        # Search for best match in a window of segments
        best_match_start = whisper_idx
        best_match_end = whisper_idx
        best_score = 0
        best_accumulated_text = ""
        
        # Try matching with 1 to max_window consecutive segments
        for start_idx in range(whisper_idx, min(whisper_idx + max_window, len(whisper_segments))):
            accumulated_text = ""
            accumulated_segments_list = []
            
            # Try different numbers of segments starting from start_idx
            for num_segments in range(1, min(max_window + 1, len(whisper_segments) - start_idx + 1)):
                if start_idx + num_segments - 1 >= len(whisper_segments):
                    break
                
                # Accumulate text from consecutive segments
                segs = []
                for i in range(num_segments):
                    seg_idx = start_idx + i
                    if seg_idx < len(whisper_segments):
                        seg = whisper_segments[seg_idx]
                        segs.append(seg)
                        if accumulated_text:
                            accumulated_text += " " + seg['text']
                        else:
                            accumulated_text = seg['text']
                
                # Calculate similarity score
                score = similarity_score_method2(accumulated_text, normalized_ref_sent)
                
                # Also check if we're getting closer to the end of the sentence
                # by comparing lengths
                ref_length = len(normalized_ref_sent)
                acc_length = len(normalize_text_for_matching_method2(accumulated_text))
                length_ratio = min(ref_length, acc_length) / max(ref_length, acc_length, 1)
                
                # Boost score if lengths are similar (we might have found the complete sentence)
                adjusted_score = score * (0.7 + 0.3 * length_ratio)
                
                # Prefer matches that are closer to the expected length
                if acc_length >= ref_length * 0.8 and acc_length <= ref_length * 1.5:
                    adjusted_score *= 1.2  # Boost if length is reasonable
                
                # Update best match if this is better
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match_start = start_idx
                    best_match_end = start_idx + num_segments - 1
                    best_accumulated_text = accumulated_text
                    accumulated_segments_list = segs
        
        # If we found a good match (threshold: 0.25 for fuzzy matching)
        if best_score > 0.25:
            # Combine segments into one
            first_seg = whisper_segments[best_match_start]
            last_seg = whisper_segments[best_match_end]
            
            # Create new combined segment
            combined_seg = {
                'id': first_seg['id'],  # Keep first segment's ID
                'start': first_seg['start'],  # Start time from first segment
                'end': last_seg['end'],  # End time from last segment
                'text': ref_sentence  # Use correct reference text
            }
            
            corrected_segments.append(combined_seg)
            
            # Move past all matched segments
            whisper_idx = best_match_end + 1
            ref_idx += 1
            
            if ref_idx % 50 == 0:
                print(f"  Matched {ref_idx}/{len(reference_sentences)} sentences (similarity: {best_score:.2%})")
        else:
            # No good match found - try to advance
            # If we've accumulated too many segments without a match, skip this reference sentence
            if whisper_idx < len(whisper_segments):
                # Keep the original segment and move forward
                seg = whisper_segments[whisper_idx].copy()
                corrected_segments.append(seg)
                whisper_idx += 1
                
                # If we've skipped too many segments, also advance reference
                if whisper_idx - best_match_start > 10:
                    ref_idx += 1
    
    # Handle remaining whisper segments
    while whisper_idx < len(whisper_segments):
        seg = whisper_segments[whisper_idx].copy()
        corrected_segments.append(seg)
        whisper_idx += 1
    
    return corrected_segments


def fix_punctuation_method2(text):
    """Fix common punctuation issues in Vietnamese text."""
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,!?:;])([^\s])', r'\1 \2', text)  # Add space after punctuation if missing
    
    # Fix quotes
    text = re.sub(r'"\s*([^"]+)\s*"', r'"\1"', text)  # Remove spaces inside quotes
    # Vietnamese quotes - using different approach to avoid quote issues
    text = re.sub(r'["\u201C\u201D]\s*([^"\u201C\u201D]+)\s*["\u201C\u201D]', r'"\1"', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def replace_whisper_transcriptions_method2(whisper_json_path, reference_text_path, output_json_path):
    """
    METHOD 2: Replace wrong whisper transcriptions with correct text from reference file.
    
    Args:
        whisper_json_path: Path to whisper_segments.json
        reference_text_path: Path to reference text file
        output_json_path: Path to output corrected JSON file
    """
    # Load whisper segments
    print(f"Loading whisper segments from {whisper_json_path}...")
    with open(whisper_json_path, 'r', encoding='utf-8') as f:
        whisper_segments = json.load(f)
    
    print(f"Loaded {len(whisper_segments)} whisper segments")
    
    # Load reference text
    print(f"Loading reference text from {reference_text_path}...")
    reference_sentences = load_text_file_method2(reference_text_path)
    print(f"Loaded {len(reference_sentences)} reference sentences")
    
    # Align and replace
    print("Aligning segments to reference text...")
    corrected_segments = align_segments_to_text_method2(whisper_segments, reference_sentences)
    
    # Fix punctuation in all segments
    print("Fixing punctuation...")
    for seg in corrected_segments:
        if seg['text']:
            seg['text'] = fix_punctuation_method2(seg['text'])
    
    # Remove empty segments
    corrected_segments = [seg for seg in corrected_segments if seg.get('text', '').strip()]
    
    # Renumber IDs sequentially
    for i, seg in enumerate(corrected_segments):
        seg['id'] = i
    
    # Save corrected segments
    print(f"Saving corrected segments to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_segments, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Corrected {len(corrected_segments)} segments (from {len(whisper_segments)} original) saved to {output_json_path}")
    print(f"Combined {len(whisper_segments) - len(corrected_segments)} segments into matching sentences")


def save_results(sentence_timestamps, transcription, output_dir):
    """
    Save timestamps and full transcription.
    
    Args:
        sentence_timestamps: List of (start, end, sentence) tuples
        transcription: Full transcription text
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save timestamps
    timestamp_file = os.path.join(output_dir, "timestamps.txt")
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        for start, end, sentence in sentence_timestamps:
            f.write(f"{start:.3f}\t{end:.3f}\t{sentence}\n")
    
    # Save full transcription
    trans_file = os.path.join(output_dir, "transcription.txt")
    with open(trans_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"Đã lưu kết quả vào: {output_dir}")
    print(f"Saved results to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Cách sử dụng / Usage:")
        print("  python align_vietnamese_audio.py <audio_file> [sentences_file] [output_dir] [model] [--use-detected] [--match-real]")
        print("  python align_vietnamese_audio.py <audio_file> [output_dir] [model] --transcribe-only")
        print("  python align_vietnamese_audio.py <whisper_dir> [sentences_file] [output_dir] --from-saved [--use-detected] [--match-real]")
        print("  python align_vietnamese_audio.py <whisper_json> <reference_text> <output_json> --method-2")
        print("\nVí dụ / Example:")
        print("  # Chỉ chạy Whisper và lưu kết quả / Only run Whisper and save results:")
        print("  python align_vietnamese_audio.py audio.wav whisper_output base --transcribe-only")
        print("  # Sử dụng kết quả đã lưu để align / Use saved results to align:")
        print("  python align_vietnamese_audio.py whisper_output sentences.txt output --from-saved")
        print("  # Sử dụng transcript có sẵn / Use provided transcript:")
        print("  python align_vietnamese_audio.py audio.wav sentences.txt output base")
        print("  # Sử dụng câu được Whisper phát hiện / Use sentences detected by Whisper:")
        print("  python align_vietnamese_audio.py audio.wav --use-detected output base")
        print("  # Khớp với câu thật từ file văn bản / Match with real sentences from text file:")
        print("  python align_vietnamese_audio.py audio.wav sentences.txt output base --use-detected --match-real")
        print("  # METHOD 2: Sửa transcript Whisper sai bằng văn bản đúng / Fix wrong Whisper transcriptions:")
        print("  python align_vietnamese_audio.py whisper_segments.json reference.txt corrected_segments.json --method-2")
        print("\nMô hình / Models: tiny, base, small, medium, large")
        print("Khuyến nghị / Recommended: base hoặc small")
        sys.exit(1)
    
    # Check for special flags
    transcribe_only = "--transcribe-only" in sys.argv
    from_saved = "--from-saved" in sys.argv
    use_detected = "--use-detected" in sys.argv
    match_real = "--match-real" in sys.argv
    method_2 = "--method-2" in sys.argv
    
    # Parse arguments - remove flags first
    args = [arg for arg in sys.argv[1:] if arg not in ["--use-detected", "--match-real", "--transcribe-only", "--from-saved", "--method-2"]]
    
    # Handle --method-2: Replace wrong whisper transcriptions with correct text
    if method_2:
        if len(args) < 3:
            print("Lỗi: Cần 3 tham số cho --method-2: <whisper_json> <reference_text> <output_json>")
            print("Error: Need 3 arguments for --method-2: <whisper_json> <reference_text> <output_json>")
            sys.exit(1)
        
        whisper_json_path = args[0]
        reference_text_path = args[1]
        output_json_path = args[2]
        
        if not os.path.exists(whisper_json_path):
            print(f"Lỗi: Không tìm thấy file JSON Whisper: {whisper_json_path}")
            print(f"Error: Whisper JSON file not found: {whisper_json_path}")
            sys.exit(1)
        
        if not os.path.exists(reference_text_path):
            print(f"Lỗi: Không tìm thấy file văn bản tham chiếu: {reference_text_path}")
            print(f"Error: Reference text file not found: {reference_text_path}")
            sys.exit(1)
        
        print("Chế độ: METHOD 2 - Sửa transcript Whisper sai bằng văn bản đúng")
        print("Mode: METHOD 2 - Fix wrong Whisper transcriptions with correct text")
        replace_whisper_transcriptions_method2(whisper_json_path, reference_text_path, output_json_path)
        return
    
    # Handle --transcribe-only: just run Whisper and save
    if transcribe_only:
        audio_file = args[0]
        whisper_output_dir = args[1] if len(args) > 1 else None
        model_name = args[2] if len(args) > 2 else "base"
        
        if not os.path.exists(audio_file):
            print(f"Lỗi: Không tìm thấy file audio: {audio_file}")
            print(f"Error: Audio file not found: {audio_file}")
            sys.exit(1)
        
        print("Chế độ: Chỉ chạy Whisper và lưu kết quả")
        print("Mode: Only run Whisper and save results")
        run_whisper_transcription(audio_file, model_name, whisper_output_dir)
        print("\n✓ Hoàn thành! Kết quả Whisper đã được lưu.")
        print("✓ Complete! Whisper results have been saved.")
        print("Bạn có thể sử dụng --from-saved để align sau.")
        print("You can use --from-saved to align later.")
        return
    
    # Handle --from-saved: load from saved Whisper results
    if from_saved:
        whisper_dir = args[0]
        
        if use_detected:
            if match_real:
                # Format: whisper_dir sentences_file [output_dir] --from-saved --use-detected --match-real
                sentences_file = args[1] if len(args) > 1 else None
                output_dir = args[2] if len(args) > 2 else "audio_segments"
            else:
                # Format: whisper_dir [output_dir] --from-saved --use-detected
                sentences_file = None
                output_dir = args[1] if len(args) > 1 else "audio_segments"
        else:
            # Format: whisper_dir sentences_file [output_dir] --from-saved
            sentences_file = args[1] if len(args) > 1 else None
            output_dir = args[2] if len(args) > 2 else "audio_segments"
        
        if not os.path.exists(whisper_dir):
            print(f"Lỗi: Không tìm thấy thư mục Whisper: {whisper_dir}")
            print(f"Error: Whisper directory not found: {whisper_dir}")
            sys.exit(1)
        
        if match_real and not sentences_file:
            print(f"Lỗi: Cần file văn bản khi sử dụng --match-real")
            print(f"Error: Sentences file required when using --match-real")
            sys.exit(1)
        
        if not use_detected and sentences_file and not os.path.exists(sentences_file):
            print(f"Lỗi: Không tìm thấy file văn bản: {sentences_file}")
            print(f"Error: Sentences file not found: {sentences_file}")
            sys.exit(1)
        
        print("Chế độ: Sử dụng kết quả Whisper đã lưu")
        print("Mode: Using saved Whisper results")
        
        # Align from saved results
        sentence_timestamps, transcription = align_from_whisper_results(
            whisper_dir, sentences_file, use_detected, match_real
        )
        
        # Save results
        save_results(sentence_timestamps, transcription, output_dir)
        
        # For cutting audio, we need the original audio file
        # Try to find it from whisper_dir name or ask user
        base_name = os.path.basename(whisper_dir).replace("_whisper_results", "")
        possible_audio = os.path.join(os.path.dirname(whisper_dir), base_name)
        # Try common extensions
        audio_file = None
        for ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            test_path = possible_audio + ext
            if os.path.exists(test_path):
                audio_file = test_path
                break
        
        if audio_file:
            cut_audio_segments(audio_file, sentence_timestamps, output_dir)
        else:
            print("⚠ Không tìm thấy file audio gốc để cắt. Vui lòng cắt thủ công.")
            print("⚠ Original audio file not found for cutting. Please cut manually.")
        
        return
    
    # Original workflow: run Whisper and align in one go
    audio_file = args[0]
    
    if use_detected:
        if match_real:
            # Format: audio_file sentences_file [output_dir] [model] --use-detected --match-real
            sentences_file = args[1] if len(args) > 1 else None
            output_dir = args[2] if len(args) > 2 else "audio_segments"
            model_name = args[3] if len(args) > 3 else "base"
        else:
            # Format: audio_file [output_dir] [model] --use-detected
            sentences_file = None
            output_dir = args[1] if len(args) > 1 else "audio_segments"
            model_name = args[2] if len(args) > 2 else "base"
    else:
        # Format: audio_file sentences_file [output_dir] [model]
        sentences_file = args[1] if len(args) > 1 else None
        output_dir = args[2] if len(args) > 2 else "audio_segments"
        model_name = args[3] if len(args) > 3 else "base"
    
    if not os.path.exists(audio_file):
        print(f"Lỗi: Không tìm thấy file audio: {audio_file}")
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    if match_real and not sentences_file:
        print(f"Lỗi: Cần file văn bản khi sử dụng --match-real")
        print(f"Error: Sentences file required when using --match-real")
        sys.exit(1)
    
    if not use_detected and sentences_file and not os.path.exists(sentences_file):
        print(f"Lỗi: Không tìm thấy file văn bản: {sentences_file}")
        print(f"Error: Sentences file not found: {sentences_file}")
        sys.exit(1)
    
    if match_real and sentences_file and not os.path.exists(sentences_file):
        print(f"Lỗi: Không tìm thấy file văn bản: {sentences_file}")
        print(f"Error: Sentences file not found: {sentences_file}")
        sys.exit(1)
    
    # Align
    sentence_timestamps, transcription = align_vietnamese_audio_improved(
        audio_file, sentences_file, model_name, 
        use_detected_sentences=use_detected,
        match_with_real_sentences=match_real
    )
    
    # Save results
    save_results(sentence_timestamps, transcription, output_dir)
    
    # Cut audio
    cut_audio_segments(audio_file, sentence_timestamps, output_dir)


if __name__ == '__main__':
    main()


