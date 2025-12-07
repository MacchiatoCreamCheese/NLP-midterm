#!/usr/bin/env python3
"""
Cut audio segments based on word_level_matches.json.

For matched sentences: uses start_time and end_time from the match.
For unmatched sentences: fills the gap between previous sentence's end_time 
and next sentence's start_time.
"""

import os
import sys
import json

# Add src directory to path to import align_vietnamese_audio functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from align_vietnamese_audio import cut_audio_segments


def load_word_matches(json_path):
    """
    Load word-level matches from JSON file.
    
    Returns:
        List of match dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    return matches


def process_matches_for_cutting(matches):
    """
    Process matches to create (start, end, sentence) tuples for audio cutting.
    
    For unmatched sentences (error/no match), fills gap between previous and next matches.
    
    Args:
        matches: List of match dictionaries from word_level_matches.json
    
    Returns:
        Tuple of (sentence_timestamps, stats_dict)
        - sentence_timestamps: List of (start, end, sentence) tuples ready for cut_audio_segments
        - stats_dict: Dictionary with statistics about processing
    """
    sentence_timestamps = []
    stats = {
        'matched_used': 0,
        'gaps_filled': 0,
        'default_duration_used': 0,
        'skipped': 0
    }
    
    for i, match in enumerate(matches):
        line_number = match.get('line_number', i + 1)
        sentence = match.get('sentence', '')
        
        # Check if this match was successful
        if match.get('matched_words') is not None and 'error' not in match:
            # Successful match - use the start_time and end_time
            start_time = match.get('start_time')
            end_time = match.get('end_time')
            
            if start_time is not None and end_time is not None:
                sentence_timestamps.append((start_time, end_time, sentence))
                stats['matched_used'] += 1
            else:
                print(f"Warning: Line {line_number} has matched_words but missing timestamps, skipping")
                stats['skipped'] += 1
        else:
            # Unmatched sentence - fill gap between previous and next
            prev_end = None
            next_start = None
            
            # Find previous match with valid end_time
            for j in range(i - 1, -1, -1):
                prev_match = matches[j]
                if (prev_match.get('matched_words') is not None and 
                    'error' not in prev_match and
                    prev_match.get('end_time') is not None):
                    prev_end = prev_match.get('end_time')
                    break
            
            # Find next match with valid start_time
            for j in range(i + 1, len(matches)):
                next_match = matches[j]
                if (next_match.get('matched_words') is not None and 
                    'error' not in next_match and
                    next_match.get('start_time') is not None):
                    next_start = next_match.get('start_time')
                    break
            
            # Determine start and end times for the gap
            if prev_end is not None and next_start is not None:
                # Fill the gap between previous end and next start
                start_time = prev_end
                end_time = next_start
                
                if start_time < end_time:
                    # Ensure minimum duration of 0.5 seconds
                    if end_time - start_time < 0.5:
                        # Expand to minimum duration, centered if possible
                        center = (start_time + end_time) / 2
                        start_time = max(0, center - 0.25)
                        end_time = center + 0.25
                    
                    sentence_timestamps.append((start_time, end_time, sentence))
                    stats['gaps_filled'] += 1
                    print(f"  Line {line_number}: Filled gap from {start_time:.2f}s to {end_time:.2f}s ({end_time-start_time:.2f}s)")
                else:
                    print(f"  Warning: Line {line_number}: Invalid gap (start >= end), skipping")
                    stats['skipped'] += 1
            elif prev_end is not None:
                # Only previous match exists - extend a small amount
                # Use a default duration of 2 seconds for unmatched sentences at the end
                start_time = prev_end
                end_time = prev_end + 2.0
                sentence_timestamps.append((start_time, end_time, sentence))
                stats['default_duration_used'] += 1
                print(f"  Line {line_number}: Using default duration 2s after {start_time:.2f}s")
            elif next_start is not None:
                # Only next match exists - use time before it
                # Use a default duration of 2 seconds ending at next_start
                end_time = next_start
                start_time = max(0, end_time - 2.0)
                sentence_timestamps.append((start_time, end_time, sentence))
                stats['default_duration_used'] += 1
                print(f"  Line {line_number}: Using default duration 2s before {end_time:.2f}s")
            else:
                # No matches around - cannot determine timing, skip
                print(f"  Warning: Line {line_number}: Cannot determine timing (no adjacent matches), skipping")
                stats['skipped'] += 1
    
    return sentence_timestamps, stats


def main():
    if len(sys.argv) < 4:
        print("Usage: python cut_audio_from_word_matches.py <word_matches_json> <audio_file> <output_dir> [padding]")
        print("Example: python cut_audio_from_word_matches.py output/word_level_matches.json data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3 output/audio_segments_method_w 0.3")
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
    
    print("=" * 80)
    print("Cutting Audio from Word-Level Matches (Method W)")
    print("=" * 80)
    print(f"Word matches file: {json_file}")
    print(f"Audio file:        {audio_file}")
    print(f"Output directory:  {output_dir}")
    print(f"Padding:           {padding} seconds")
    print()
    
    # Load matches
    print(f"Loading word matches from {json_file}...")
    matches = load_word_matches(json_file)
    print(f"Loaded {len(matches)} sentence matches")
    
    # Count matched vs unmatched
    matched_count = sum(1 for m in matches if m.get('matched_words') is not None and 'error' not in m)
    unmatched_count = len(matches) - matched_count
    print(f"  Matched:   {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print()
    
    # Process matches for cutting
    print("Processing matches for audio cutting...")
    sentence_timestamps, stats = process_matches_for_cutting(matches)
    print()
    
    if len(sentence_timestamps) == 0:
        print("Error: No valid segments to cut")
        sys.exit(1)
    
    # Validate timestamps (check for overlaps and order)
    print("Validating timestamps...")
    invalid_count = 0
    for i, (start, end, text) in enumerate(sentence_timestamps):
        if start >= end:
            print(f"  Warning: Segment {i+1} has invalid timestamps (start >= end): {start:.2f}s - {end:.2f}s")
            invalid_count += 1
        if i > 0:
            prev_start, prev_end, _ = sentence_timestamps[i-1]
            if start < prev_end:
                print(f"  Warning: Segment {i+1} overlaps with previous: {prev_end:.2f}s vs {start:.2f}s")
                invalid_count += 1
    
    if invalid_count > 0:
        print(f"  Found {invalid_count} potential issues")
    else:
        print("  All timestamps are valid")
    
    print(f"\nCreated {len(sentence_timestamps)} audio segments to cut")
    print()
    
    # Show first few segments for verification
    print("First 3 segments:")
    for i, (start, end, text) in enumerate(sentence_timestamps[:3]):
        print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s ({end-start:.3f}s) - {text[:50]}...")
    print()
    
    # Cut audio
    print(f"Cutting audio from {audio_file}...")
    print(f"Output directory: {output_dir}")
    print(f"Padding: {padding} seconds (added before and after each segment)")
    print()
    
    cut_audio_segments(audio_file, sentence_timestamps, output_dir, add_padding=padding)
    
    print()
    print("=" * 80)
    print("Processing Summary:")
    print(f"  Segments from matched sentences: {stats['matched_used']}")
    print(f"  Segments from filled gaps:       {stats['gaps_filled']}")
    print(f"  Segments with default duration:  {stats['default_duration_used']}")
    print(f"  Skipped (no timing available):   {stats['skipped']}")
    print(f"  Total segments created:          {len(sentence_timestamps)}")
    print()
    print("=" * 80)
    print(f"✓ Complete! Audio segments saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

