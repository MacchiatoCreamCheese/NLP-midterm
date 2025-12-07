#!/usr/bin/env python3
"""
Script to cut audio segments from JSON files containing segment timestamps.
Usage: python cut_audio_from_json.py <json_file> <audio_file> <output_dir>
"""

import os
import sys
import json

# Add src directory to path to import align_vietnamese_audio functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from align_vietnamese_audio import cut_audio_segments


def load_segments_from_json(json_path):
    """
    Load segments from JSON file and convert to format expected by cut_audio_segments.
    
    Args:
        json_path: Path to JSON file with segments
        
    Returns:
        List of (start, end, sentence) tuples
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    sentence_timestamps = []
    for i, seg in enumerate(segments):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        
        # Validate timestamps
        if start < 0:
            print(f"Warning: Segment {i} has negative start time, setting to 0")
            start = 0
        if end <= start:
            print(f"Warning: Segment {i} has invalid end time ({end} <= {start}), skipping")
            continue
        
        if text:
            sentence_timestamps.append((start, end, text))
        else:
            print(f"Warning: Segment {i} has no text, skipping")
    
    # Sort by start time to ensure correct order
    sentence_timestamps.sort(key=lambda x: x[0])
    
    return sentence_timestamps


def main():
    if len(sys.argv) < 4:
        print("Usage: python cut_audio_from_json.py <json_file> <audio_file> <output_dir> [padding]")
        print("Example: python cut_audio_from_json.py output/corrected_segments.json data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3 output/audio_segments_method2")
        print("Optional: padding (default: 0.3 seconds)")
        sys.exit(1)
    
    json_file = sys.argv[1]
    audio_file = sys.argv[2]
    output_dir = sys.argv[3]
    padding = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print(f"Loading segments from {json_file}...")
    sentence_timestamps = load_segments_from_json(json_file)
    print(f"Loaded {len(sentence_timestamps)} segments")
    
    if len(sentence_timestamps) == 0:
        print("Error: No valid segments found in JSON file")
        sys.exit(1)
    
    # Show first few segments for verification
    print("\nFirst 3 segments:")
    for i, (start, end, text) in enumerate(sentence_timestamps[:3]):
        print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s ({end-start:.3f}s) - {text[:50]}...")
    
    print(f"\nCutting audio from {audio_file}...")
    print(f"Output directory: {output_dir}")
    print(f"Padding: {padding} seconds (added before and after each segment)")
    
    cut_audio_segments(audio_file, sentence_timestamps, output_dir, add_padding=padding)
    
    print(f"\n✓ Complete! Audio segments saved to: {output_dir}")


if __name__ == '__main__':
    main()

