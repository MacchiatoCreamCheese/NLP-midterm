#!/usr/bin/env python3
"""
Diagnostic script to check if audio cutting matches JSON timestamps.
"""

import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def check_audio_cutting(json_file, audio_file, segment_index=7):
    """Check if a specific segment is cut correctly."""
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    if segment_index >= len(segments):
        print(f"Error: Segment index {segment_index} out of range (max: {len(segments)-1})")
        return
    
    seg = segments[segment_index]
    start_time = seg['start']
    end_time = seg['end']
    text = seg['text']
    
    print(f"JSON Segment {segment_index}:")
    print(f"  Start: {start_time:.3f}s")
    print(f"  End: {end_time:.3f}s")
    print(f"  Duration: {end_time - start_time:.3f}s")
    print(f"  Text: {text[:60]}...")
    
    if not HAS_LIBROSA:
        print("\nError: librosa not available for testing")
        return
    
    # Load audio
    print(f"\nLoading audio: {audio_file}")
    audio_array, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    print(f"Audio loaded: {len(audio_array)} samples @ {sample_rate}Hz")
    print(f"Total duration: {len(audio_array)/sample_rate:.2f}s")
    
    # Calculate expected sample indices (with 0.3s padding)
    padding = 0.3
    start_sample = max(0, int((start_time - padding) * sample_rate))
    end_sample = min(len(audio_array), int((end_time + padding) * sample_rate))
    
    print(f"\nCalculated sample indices (with {padding}s padding):")
    print(f"  Start sample: {start_sample} ({start_sample/sample_rate:.3f}s)")
    print(f"  End sample: {end_sample} ({end_sample/sample_rate:.3f}s)")
    print(f"  Duration samples: {end_sample - start_sample} ({(end_sample - start_sample)/sample_rate:.3f}s)")
    
    # Extract segment
    segment = audio_array[start_sample:end_sample]
    print(f"\nExtracted segment:")
    print(f"  Length: {len(segment)} samples")
    print(f"  Duration: {len(segment)/sample_rate:.3f}s")
    print(f"  Expected duration: {(end_time - start_time + 2*padding):.3f}s")
    
    # Check if there's actual audio (non-silent)
    if len(segment) > 0:
        max_amplitude = np.max(np.abs(segment))
        print(f"  Max amplitude: {max_amplitude:.4f}")
        if max_amplitude < 0.01:
            print("  WARNING: Segment appears to be mostly silent!")
    
    # Check surrounding segments for context
    if segment_index > 0:
        prev_seg = segments[segment_index - 1]
        print(f"\nPrevious segment (index {segment_index - 1}):")
        print(f"  End: {prev_seg['end']:.3f}s")
        print(f"  Gap: {start_time - prev_seg['end']:.3f}s")
    
    if segment_index < len(segments) - 1:
        next_seg = segments[segment_index + 1]
        print(f"\nNext segment (index {segment_index + 1}):")
        print(f"  Start: {next_seg['start']:.3f}s")
        print(f"  Gap: {next_seg['start'] - end_time:.3f}s")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python diagnose_audio_cutting.py <json_file> <audio_file> [segment_index]")
        print("Example: python diagnose_audio_cutting.py output/corrected_segments.json data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3 7")
        sys.exit(1)
    
    json_file = sys.argv[1]
    audio_file = sys.argv[2]
    segment_index = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    
    check_audio_cutting(json_file, audio_file, segment_index)

