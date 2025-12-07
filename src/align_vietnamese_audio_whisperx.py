#!/usr/bin/env python3
"""
Vietnamese audio-text alignment using WhisperX for better forced alignment.
WhisperX provides more accurate word-level timestamps than basic Whisper.
"""

import os
import sys
import re
import numpy as np
import librosa
import soundfile as sf
from difflib import SequenceMatcher
import unicodedata

try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False
    print("Warning: whisperx not installed. Install with: pip install whisperx")
    print("Falling back to basic Whisper alignment...")

# Fix for PyTorch 2.6 compatibility with whisperx/pyannote
# This must be done BEFORE importing whisperx to ensure the patch is active
try:
    import torch
    
    # Add safe globals for omegaconf classes that appear in pyannote model checkpoints
    if hasattr(torch.serialization, 'add_safe_globals'):
        safe_globals_list = []
        
        # Try to import and add all known omegaconf classes
        omegaconf_classes = [
            ('omegaconf.listconfig', 'ListConfig'),
            ('omegaconf.base', 'ContainerMetadata'),
            ('omegaconf.base', 'DictConfig'),
            ('omegaconf.base', 'ListConfig'),
            ('omegaconf.dictconfig', 'DictConfig'),
        ]
        
        for module_path, class_name in omegaconf_classes:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name, None)
                if cls and cls not in safe_globals_list:
                    safe_globals_list.append(cls)
            except (ImportError, AttributeError):
                pass
        
        # Also try to dynamically discover omegaconf classes
        try:
            import omegaconf
            for attr_name in dir(omegaconf):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(omegaconf, attr_name)
                        if isinstance(attr, type) and 'omegaconf' in str(type(attr)):
                            if attr not in safe_globals_list:
                                safe_globals_list.append(attr)
                    except:
                        pass
        except:
            pass
        
        if safe_globals_list:
            torch.serialization.add_safe_globals(safe_globals_list)
    
    # Monkey-patch torch.load to always use weights_only=False for compatibility
    # PyTorch 2.6 changed default from False to True, breaking pyannote/whisperx
    _original_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        # Force weights_only=False for all loads (trusted model files from HuggingFace)
        kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load
except (ImportError, AttributeError):
    # If torch not available, skip patching
    pass


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
        whisper_segments: List of (start, end, text) tuples from WhisperX
        real_sentences: List of real sentences from text file
        word_segments: List of word dicts with 'word', 'start', 'end' from WhisperX
    
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


def align_sentences_with_whisperx_words(sentences, word_segments):
    """
    Align sentences with WhisperX word-level timestamps.
    WhisperX provides more accurate word boundaries than basic Whisper.
    
    Args:
        sentences: List of Vietnamese sentences
        word_segments: List of word segments from WhisperX with 'word', 'start', 'end'
    
    Returns:
        List of (start_time, end_time, sentence) tuples
    """
    sentence_timestamps = []
    word_idx = 0
    failed_alignments = []
    
    for i, sentence in enumerate(sentences):
        if word_idx >= len(word_segments):
            # Estimate remaining sentences
            if sentence_timestamps:
                last_end = sentence_timestamps[-1][1]
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((last_end, last_end + estimated_duration, sentence))
            continue
        
        # Find sentence in word segments
        sentence_normalized = normalize_text(sentence)
        sentence_words = sentence_normalized.split()
        
        if not sentence_words:
            continue
        
        # Try to find matching words
        best_start_idx = None
        best_end_idx = None
        best_score = 0
        
        # Search for sentence in word segments
        for start_idx in range(word_idx, min(word_idx + 200, len(word_segments) - len(sentence_words) + 1)):
            matched = 0
            trans_idx = start_idx
            
            for sent_word in sentence_words[:20]:  # Check first 20 words
                if trans_idx >= len(word_segments):
                    break
                
                trans_word = normalize_text(word_segments[trans_idx]['word'])
                
                # Exact match
                if sent_word == trans_word:
                    matched += 1
                    trans_idx += 1
                # Partial match
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
                    break
            
            score = matched / len(sentence_words[:20])
            if score > best_score:
                best_score = score
                best_start_idx = start_idx
                best_end_idx = trans_idx
        
        # Use alignment if confidence is good
        if best_score > 0.4 and best_start_idx is not None and best_end_idx is not None:
            start_time = word_segments[best_start_idx]['start']
            end_time = word_segments[best_end_idx - 1]['end'] if best_end_idx > 0 else word_segments[best_start_idx]['end']
            sentence_timestamps.append((start_time, end_time, sentence))
            word_idx = best_end_idx
            
            if (i + 1) % 100 == 0:
                print(f"  Đã căn chỉnh {i+1}/{len(sentences)} câu (độ tin cậy: {best_score:.2%})")
                print(f"  Aligned {i+1}/{len(sentences)} sentences (confidence: {best_score:.2%})")
        else:
            # Fallback: estimate
            if sentence_timestamps:
                last_end = sentence_timestamps[-1][1]
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((last_end, last_end + estimated_duration, sentence))
                failed_alignments.append(i)
            else:
                start_time = word_segments[word_idx]['start'] if word_segments else 0
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((start_time, start_time + estimated_duration, sentence))
                failed_alignments.append(i)
            
            word_idx += len(sentence_words)
    
    if failed_alignments:
        print(f"\nCảnh báo: {len(failed_alignments)} câu không thể căn chỉnh chính xác (đã ước tính)")
        print(f"Warning: {len(failed_alignments)} sentences could not be aligned (estimated)")
    
    return sentence_timestamps


def align_vietnamese_audio_whisperx(audio_file, sentences_file=None, model_name="base", device="cpu", use_detected_sentences=False, match_with_real_sentences=False):
    """
    Align Vietnamese audio with text using WhisperX for better accuracy.
    
    Args:
        audio_file: Path to audio file
        sentences_file: Path to file with Vietnamese sentences (one per line). Optional if use_detected_sentences=True.
        model_name: Whisper model to use (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda)
        use_detected_sentences: If True, use sentences detected by WhisperX instead of provided transcript
    
    Returns:
        Tuple of (sentence_timestamps, full_transcription)
    """
    if not HAS_WHISPERX:
        raise RuntimeError("WhisperX is not installed. Install with: pip install whisperx")
    
    print(f"Đang tải audio bằng librosa...")
    print(f"Loading audio with librosa...")
    sys.stdout.flush()
    
    # Load audio with librosa
    audio_array, sample_rate = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)
    
    if len(audio_array) == 0:
        raise RuntimeError("Loaded audio array is empty")
    
    print(f"✓ Đã tải audio: {len(audio_array)} samples @ {sample_rate}Hz")
    print(f"✓ Loaded audio: {len(audio_array)} samples @ {sample_rate}Hz")
    
    print(f"Đang tải mô hình WhisperX: {model_name}...")
    print(f"Loading WhisperX model: {model_name}...")
    sys.stdout.flush()
    
    # Load WhisperX model
    # Use float32 for CPU (float16 not supported efficiently on CPU)
    compute_type = "float32" if device == "cpu" else "float16"
    model = whisperx.load_model(model_name, device=device, language="vi", compute_type=compute_type)
    
    print("Đang chuyển đổi giọng nói sang văn bản với WhisperX...")
    print("Transcribing audio with WhisperX...")
    sys.stdout.flush()
    
    # Transcribe with WhisperX
    result = model.transcribe(audio_array, batch_size=16)
    
    print("Đang căn chỉnh từ với mô hình alignment...")
    print("Aligning words with alignment model...")
    sys.stdout.flush()
    
    # Load alignment model for Vietnamese
    try:
        align_model, metadata = whisperx.load_align_model(language_code="vi", device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio_array, device=device, return_char_alignments=False)
    except Exception as e:
        print(f"⚠ Không thể tải mô hình alignment, sử dụng timestamps từ WhisperX: {e}")
        print(f"⚠ Could not load alignment model, using WhisperX timestamps: {e}")
    
    # Get full transcription
    full_transcription = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
    
    # If using detected sentences, match with real sentences first, then cut audio
    if use_detected_sentences:
        print("Đang sử dụng các câu được phát hiện bởi WhisperX...")
        print("Using sentences detected by WhisperX...")
        sys.stdout.flush()
        
        # Extract all word timestamps
        all_word_segments = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                all_word_segments.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0)
                })
        
        # Extract raw WhisperX segments (without merging/splitting)
        raw_whisper_segments = []
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if text:
                raw_whisper_segments.append((
                    segment.get("start", 0),
                    segment.get("end", 0),
                    text
                ))
        
        # If match_with_real_sentences is True and sentences_file is provided, match with real sentences FIRST
        if match_with_real_sentences and sentences_file:
            print("\nĐang khớp các câu thật với transcript WhisperX...")
            print("Matching real sentences with WhisperX transcript...")
            sys.stdout.flush()
            
            # Read real sentences
            with open(sentences_file, 'r', encoding='utf-8') as f:
                real_sentences = [line.strip() for line in f if line.strip()]
            
            # Match each whole real sentence to WhisperX segments (looser matching)
            sentence_timestamps = match_whisper_to_real_sentences(
                raw_whisper_segments, real_sentences, all_word_segments
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
            
            for segment in result.get("segments", []):
                text = segment.get("text", "").strip()
                if not text:
                    continue
                
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                
                # Get words for this segment
                segment_words = []
                for word_info in segment.get("words", []):
                    segment_words.append({
                        "word": word_info.get("word", "").strip(),
                        "start": word_info.get("start", 0),
                        "end": word_info.get("end", 0)
                    })
                
                # Merge with current accumulated text
                if current_merged_text:
                    current_merged_text += " " + text
                else:
                    current_merged_text = text
                    current_merged_start = segment_start
                
                current_merged_end = segment_end
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
    
    # Extract word-level timestamps from WhisperX result
    word_segments = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_segments.append({
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0)
            })
    
    print(f"Đã nhận diện {len(word_segments)} từ trong audio")
    print(f"Recognized {len(word_segments)} words in audio")
    print(f"Đang căn chỉnh {len(sentences)} câu với audio...")
    print(f"Aligning {len(sentences)} sentences with audio...")
    sys.stdout.flush()
    
    # Align sentences
    sentence_timestamps = align_sentences_with_whisperx_words(sentences, word_segments)
    
    return sentence_timestamps, full_transcription


def cut_audio_segments(audio_file, sentence_timestamps, output_dir, add_padding=0.3):
    """
    Cut audio into segments with padding using librosa and soundfile.
    """
    print(f"Đang tải audio: {audio_file}...")
    print(f"Loading audio: {audio_file}...")
    sys.stdout.flush()
    
    # Load audio with librosa
    audio_array, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Đang cắt {len(sentence_timestamps)} đoạn audio...")
    print(f"Cutting {len(sentence_timestamps)} audio segments...")
    sys.stdout.flush()
    
    for i, (start, end, sentence) in enumerate(sentence_timestamps):
        # Calculate sample indices with padding
        start_sample = max(0, int((start - add_padding) * sample_rate))
        end_sample = min(len(audio_array), int((end + add_padding) * sample_rate))
        
        # Extract segment
        segment = audio_array[start_sample:end_sample]
        
        output_file = os.path.join(output_dir, f"sentence_{i+1:05d}.wav")
        
        # Save segment using soundfile
        sf.write(output_file, segment, sample_rate)
        
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


def save_results(sentence_timestamps, transcription, output_dir):
    """Save timestamps and full transcription."""
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
        print("  python align_vietnamese_audio_whisperx.py <audio_file> [sentences_file] [output_dir] [model] [device] [--use-detected] [--match-real]")
        print("\nVí dụ / Example:")
        print("  # Sử dụng transcript có sẵn / Use provided transcript:")
        print("  python align_vietnamese_audio_whisperx.py audio.wav sentences.txt output base cpu")
        print("  # Sử dụng câu được WhisperX phát hiện / Use sentences detected by WhisperX:")
        print("  python align_vietnamese_audio_whisperx.py audio.wav --use-detected output base cpu")
        print("  # Khớp với câu thật từ file văn bản / Match with real sentences from text file:")
        print("  python align_vietnamese_audio_whisperx.py audio.wav sentences.txt output base cpu --use-detected --match-real")
        print("\nMô hình / Models: tiny, base, small, medium, large")
        print("Thiết bị / Device: cpu, cuda (nếu có GPU)")
        print("Khuyến nghị / Recommended: base hoặc small")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Check for flags
    use_detected = "--use-detected" in sys.argv
    match_real = "--match-real" in sys.argv
    
    # Parse arguments - remove flags first
    args = [arg for arg in sys.argv[1:] if arg not in ["--use-detected", "--match-real"]]
    
    if use_detected:
        if match_real:
            # Format: audio_file sentences_file [output_dir] [model] [device] --use-detected --match-real
            sentences_file = args[1] if len(args) > 1 else None
            output_dir = args[2] if len(args) > 2 else "audio_segments"
            model_name = args[3] if len(args) > 3 else "base"
            device = args[4] if len(args) > 4 else "cpu"
        else:
            # Format: audio_file [output_dir] [model] [device] --use-detected
            sentences_file = None
            output_dir = args[1] if len(args) > 1 else "audio_segments"
            model_name = args[2] if len(args) > 2 else "base"
            device = args[3] if len(args) > 3 else "cpu"
    else:
        # Format: audio_file sentences_file [output_dir] [model] [device]
        sentences_file = args[1] if len(args) > 1 else None
        output_dir = args[2] if len(args) > 2 else "audio_segments"
        model_name = args[3] if len(args) > 3 else "base"
        device = args[4] if len(args) > 4 else "cpu"
    
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
    sentence_timestamps, transcription = align_vietnamese_audio_whisperx(
        audio_file, sentences_file, model_name, device, 
        use_detected_sentences=use_detected,
        match_with_real_sentences=match_real
    )
    
    # Save results
    save_results(sentence_timestamps, transcription, output_dir)
    
    # Cut audio
    cut_audio_segments(audio_file, sentence_timestamps, output_dir)


if __name__ == '__main__':
    main()

