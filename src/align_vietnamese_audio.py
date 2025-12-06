#!/usr/bin/env python3
"""
Improved Vietnamese audio-text alignment using Whisper with better matching strategies.
Aligns Vietnamese audio with Vietnamese text files and cuts audio into sentence segments.
"""

import os
import sys
import re
from pydub import AudioSegment
import whisper
from difflib import SequenceMatcher
import unicodedata
import numpy as np
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available, will try other methods")


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


def align_vietnamese_audio_improved(audio_file, sentences_file, model_name="base"):
    """
    Improved alignment for Vietnamese audio and text.
    
    Args:
        audio_file: Path to audio file
        sentences_file: Path to file with Vietnamese sentences (one per line)
        model_name: Whisper model to use
    
    Returns:
        Tuple of (sentence_timestamps, full_transcription)
    """
    # Convert audio to WAV format using pydub (avoids ffmpeg issues with Whisper)
    import tempfile
    import subprocess
    temp_wav = None
    audio_file_for_whisper = audio_file
    conversion_success = False
    
    # Method 1: Try pydub
    try:
        print("Đang chuyển đổi audio sang định dạng WAV (phương pháp 1: pydub)...")
        print("Converting audio to WAV format (method 1: pydub)...")
        sys.stdout.flush()  # Force output
        
        # Try to load with explicit format
        try:
            if audio_file.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(audio_file)
            else:
                audio = AudioSegment.from_file(audio_file)
        except Exception as load_error:
            print(f"Load error: {load_error}, trying generic loader...")
            audio = AudioSegment.from_file(audio_file)
        
        # Export to temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        audio.export(temp_wav.name, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        
        # Verify file was created
        if os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 0:
            audio_file_for_whisper = temp_wav.name
            conversion_success = True
            print(f"✓ Đã chuyển đổi thành công: {temp_wav.name}")
            print(f"✓ Successfully converted: {temp_wav.name}")
        else:
            raise Exception("Converted file is empty or doesn't exist")
            
    except Exception as e:
        print(f"⚠ Phương pháp 1 thất bại: {type(e).__name__}: {e}")
        print(f"⚠ Method 1 failed: {type(e).__name__}: {e}")
        sys.stdout.flush()
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except:
                pass
        temp_wav = None
    
    # Method 2: Try ffmpeg directly if pydub failed
    if not conversion_success:
        try:
            print("Đang thử phương pháp 2: ffmpeg trực tiếp...")
            print("Trying method 2: direct ffmpeg...")
            sys.stdout.flush()
            
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Use ffmpeg command directly
            cmd = [
                'ffmpeg', '-i', audio_file,
                '-ar', '16000', '-ac', '1', '-f', 'wav',
                '-y', temp_wav.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 0:
                audio_file_for_whisper = temp_wav.name
                conversion_success = True
                print(f"✓ Đã chuyển đổi thành công bằng ffmpeg: {temp_wav.name}")
                print(f"✓ Successfully converted using ffmpeg: {temp_wav.name}")
            else:
                raise Exception(f"ffmpeg failed with return code {result.returncode}: {result.stderr[:200]}")
        except Exception as e:
            print(f"⚠ Phương pháp 2 thất bại: {type(e).__name__}: {e}")
            print(f"⚠ Method 2 failed: {type(e).__name__}: {e}")
            sys.stdout.flush()
            if temp_wav and os.path.exists(temp_wav.name):
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass
            temp_wav = None
    
    # Method 3: Use librosa to load audio directly (bypasses ffmpeg)
    audio_array = None
    if not conversion_success and HAS_LIBROSA:
        try:
            print("Đang thử phương pháp 3: librosa (bỏ qua ffmpeg)...")
            print("Trying method 3: librosa (bypass ffmpeg)...")
            sys.stdout.flush()
            
            # Load audio with librosa (16kHz mono, as Whisper expects)
            # dtype=np.float32 ensures compatibility with Whisper
            audio_array, sr = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)
            
            # Verify the audio array format
            if len(audio_array) == 0:
                raise Exception("Loaded audio array is empty")
            
            conversion_success = True
            print(f"✓ Đã tải audio bằng librosa: {len(audio_array)} samples @ {sr}Hz")
            print(f"✓ Successfully loaded audio with librosa: {len(audio_array)} samples @ {sr}Hz")
        except Exception as e:
            print(f"⚠ Phương pháp 3 thất bại: {type(e).__name__}: {e}")
            print(f"⚠ Method 3 failed: {type(e).__name__}: {e}")
            sys.stdout.flush()
    
    # If all conversion methods failed, warn but continue
    if not conversion_success:
        print("⚠ Cảnh báo: Không thể chuyển đổi audio, thử dùng file gốc...")
        print("⚠ Warning: Could not convert audio, trying original file...")
        print("⚠ Lưu ý: Whisper có thể gặp lỗi nếu ffmpeg không hoạt động đúng")
        print("⚠ Note: Whisper may fail if ffmpeg is not working properly")
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
        # Clean up temporary file before re-raising
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except:
                pass
        print(f"\n❌ Lỗi khi chuyển đổi giọng nói: {type(e).__name__}: {e}")
        print(f"❌ Error during transcription: {type(e).__name__}: {e}")
        sys.stdout.flush()
        raise RuntimeError(f"Whisper transcription failed. This is likely due to ffmpeg issues. "
                          f"Please ensure ffmpeg is properly installed. Error: {e}") from e
    
    # Clean up temporary file
    if temp_wav and os.path.exists(temp_wav.name):
        try:
            os.unlink(temp_wav.name)
        except:
            pass
    
    # Read sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Extract word timestamps
    words = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })
    
    print(f"Đã nhận diện {len(words)} từ trong audio")
    print(f"Recognized {len(words)} words in audio")
    print(f"Đang căn chỉnh {len(sentences)} câu với audio...")
    print(f"Aligning {len(sentences)} sentences with audio...")
    
    # Align sentences
    sentence_timestamps = []
    word_idx = 0
    failed_alignments = []
    
    for i, sentence in enumerate(sentences):
        if word_idx >= len(words):
            # Estimate remaining sentences
            if sentence_timestamps:
                last_end = sentence_timestamps[-1][1]
                estimated_duration = len(sentence.split()) / 3.5  # ~3.5 words/sec
                sentence_timestamps.append((last_end, last_end + estimated_duration, sentence))
            continue
        
        # Find sentence in transcription
        start_idx, end_idx, confidence = find_sentence_in_transcription(
            sentence, words, word_idx
        )
        
        if start_idx is not None and end_idx is not None:
            start_time = words[start_idx]['start']
            end_time = words[end_idx - 1]['end'] if end_idx > 0 else words[start_idx]['end']
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
                start_time = words[word_idx]['start'] if words else 0
                estimated_duration = len(sentence.split()) / 3.5
                sentence_timestamps.append((start_time, start_time + estimated_duration, sentence))
                failed_alignments.append(i)
            
            word_idx += len(sentence.split())  # Skip some words
    
    if failed_alignments:
        print(f"\nCảnh báo: {len(failed_alignments)} câu không thể căn chỉnh chính xác (đã ước tính)")
        print(f"Warning: {len(failed_alignments)} sentences could not be aligned (estimated)")
    
    return sentence_timestamps, result["text"]


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
            print(f"⚠ Librosa failed: {e}, trying pydub...")
            print(f"⚠ Librosa failed: {e}, trying pydub...")
            audio_array = None
    
    # Fallback to pydub if librosa failed
    if audio_array is None:
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
                    temp_seg = AudioSegment(
                        segment.tobytes(),
                        frame_rate=int(sample_rate),
                        channels=1,
                        sample_width=2
                    )
                    temp_seg.export(output_file, format="wav")
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
    if len(sys.argv) < 3:
        print("Cách sử dụng / Usage:")
        print("  python align_vietnamese_audio.py <audio_file> <sentences_file> [output_dir] [model]")
        print("\nVí dụ / Example:")
        print("  python align_vietnamese_audio.py audio.wav sentences.txt output base")
        print("\nMô hình / Models: tiny, base, small, medium, large")
        print("Khuyến nghị / Recommended: base hoặc small")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    sentences_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "audio_segments"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "base"
    
    if not os.path.exists(audio_file):
        print(f"Lỗi: Không tìm thấy file audio: {audio_file}")
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    if not os.path.exists(sentences_file):
        print(f"Lỗi: Không tìm thấy file văn bản: {sentences_file}")
        print(f"Error: Sentences file not found: {sentences_file}")
        sys.exit(1)
    
    # Align
    sentence_timestamps, transcription = align_vietnamese_audio_improved(
        audio_file, sentences_file, model_name
    )
    
    # Save results
    save_results(sentence_timestamps, transcription, output_dir)
    
    # Cut audio
    cut_audio_segments(audio_file, sentence_timestamps, output_dir)


if __name__ == '__main__':
    main()


