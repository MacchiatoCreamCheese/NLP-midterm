#!/usr/bin/env python3
"""
Split JSON word-level match files and audio files by chapter boundaries.

For each Track JSON file, splits at specified chapter boundaries:
- Track_1: Split at Chuong2 (line_number 100)
- Track_2: Split at Chuong4 (line_number 203)
- Track_3: Split at Chuong6 (line_number 264)
- Track_4: Split at Chuong7 (line_number 1), Chuong8 (line_number 185), Chuong9 (line_number 365)
- Track_5: Split at Chuong11 (line_number 130)

Process:
1. Find split points by matching first sentences from chapter text files
2. Cut audio files using ORIGINAL timestamps
3. Split JSON files and update timestamps (subtract original start_time, add padding)
4. Save new files (original files remain unchanged)
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import librosa
import soundfile as sf


def normalize_sentence(sentence: str) -> str:
    """
    Normalize sentence text for comparison.
    - Convert to lowercase
    - Remove extra whitespace
    - Handle punctuation differences
    """
    # Convert to lowercase and strip
    normalized = sentence.lower().strip()
    # Remove multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove trailing punctuation (but keep internal punctuation)
    normalized = normalized.rstrip('.,;:!?')
    return normalized


def find_split_point(json_data: List[Dict[str, Any]], target_sentence: str) -> Optional[Tuple[int, float]]:
    """
    Find the line_number and original start_time for a matching sentence in JSON data.
    
    Args:
        json_data: List of match dictionaries from word_level_matches.json
        target_sentence: The sentence text to find (first sentence from chapter)
    
    Returns:
        Tuple of (line_number, start_time) if found, None otherwise
    """
    normalized_target = normalize_sentence(target_sentence)
    
    def get_start_time(entry: Dict[str, Any], entry_idx: int) -> Optional[float]:
        """Extract start_time from entry, or estimate from surrounding entries."""
        # Try direct start_time
        start_time = entry.get('start_time')
        if start_time is not None:
            return start_time
        
        # Try from matched_words
        matched_words = entry.get('matched_words')
        if matched_words and len(matched_words) > 0:
            first_word = matched_words[0]
            if isinstance(first_word, dict) and 'start' in first_word:
                return first_word['start']
        
        # Estimate from previous entry's end_time
        if entry_idx > 0:
            prev_entry = json_data[entry_idx - 1]
            prev_end = prev_entry.get('end_time')
            if prev_end is not None:
                return prev_end
            
            # Try from previous entry's matched_words
            prev_matched = prev_entry.get('matched_words')
            if prev_matched and len(prev_matched) > 0:
                last_word = prev_matched[-1]
                if isinstance(last_word, dict) and 'end' in last_word:
                    return last_word['end']
        
        # Estimate from next entry's start_time
        if entry_idx < len(json_data) - 1:
            next_entry = json_data[entry_idx + 1]
            next_start = next_entry.get('start_time')
            if next_start is not None:
                # Use a small gap before next entry
                return max(0.0, next_start - 2.0)
            
            # Try from next entry's matched_words
            next_matched = next_entry.get('matched_words')
            if next_matched and len(next_matched) > 0:
                first_word = next_matched[0]
                if isinstance(first_word, dict) and 'start' in first_word:
                    return max(0.0, first_word['start'] - 2.0)
        
        return None
    
    # First pass: exact match
    for idx, entry in enumerate(json_data):
        sentence = entry.get('sentence', '')
        if not sentence:
            continue
            
        normalized_entry = normalize_sentence(sentence)
        
        if normalized_entry == normalized_target:
            line_number = entry.get('line_number')
            if line_number is not None:
                start_time = get_start_time(entry, idx)
                if start_time is not None:
                    if entry.get('matched_words') is None:
                        print(f"  Warning: Line {line_number} has no match, estimated start_time: {start_time:.2f}s")
                    return (line_number, start_time)
                else:
                    print(f"  Warning: Could not determine start_time for line {line_number}")
                    return None
    
    # Second pass: fuzzy matching (substring match)
    for idx, entry in enumerate(json_data):
        sentence = entry.get('sentence', '')
        if not sentence:
            continue
            
        normalized_entry = normalize_sentence(sentence)
        
        # Check if target is contained in entry or vice versa (with minimum length)
        if len(normalized_target) > 20 and len(normalized_entry) > 20:
            if normalized_target in normalized_entry or normalized_entry in normalized_target:
                line_number = entry.get('line_number')
                if line_number is not None:
                    start_time = get_start_time(entry, idx)
                    if start_time is not None:
                        print(f"  Warning: Using fuzzy match for line {line_number}: {sentence[:50]}...")
                        return (line_number, start_time)
    
    return None


def cut_audio_segment(
    audio_file: Path,
    original_start_time: float,
    original_end_time: float,
    output_file: Path,
    padding: float = 0.0
) -> None:
    """
    Extract audio segment using ORIGINAL timestamps.
    
    Args:
        audio_file: Path to original audio file
        original_start_time: Start time in seconds (from original JSON)
        original_end_time: End time in seconds (from original JSON)
        output_file: Path to save extracted audio segment
        padding: Additional padding to add before start and after end (default: 0.0)
    """
    # Load audio
    audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
    duration = len(audio) / sr
    
    # Validate timestamps
    if original_start_time < 0:
        original_start_time = 0.0
    if original_end_time <= original_start_time:
        print(f"  Error: Invalid timestamps (start={original_start_time:.2f}s, end={original_end_time:.2f}s)")
        print(f"  Using audio duration as end_time: {duration:.2f}s")
        original_end_time = duration
    
    # Calculate cut times with padding
    start_time = max(0.0, original_start_time - padding)
    end_time = min(duration, original_end_time + padding)
    
    # Ensure end_time > start_time
    if end_time <= start_time:
        end_time = min(duration, start_time + 1.0)  # At least 1 second
    
    # Convert to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Ensure valid sample range
    start_sample = max(0, min(start_sample, len(audio) - 1))
    end_sample = max(start_sample + 1, min(end_sample, len(audio)))
    
    # Extract segment
    segment = audio[start_sample:end_sample]
    
    if len(segment) == 0:
        raise ValueError(f"Empty audio segment: start={start_time:.2f}s, end={end_time:.2f}s")
    
    # Save audio
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_file), segment, sr)
    
    print(f"  Cut audio: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s) -> {output_file.name}")


def update_timestamps(
    json_part: List[Dict[str, Any]],
    original_start_time: float,
    padding: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Update all timestamps in JSON part.
    - Subtract original_start_time from all times
    - Add padding to the first sentence's start time
    
    Args:
        json_part: List of match dictionaries for this part
        original_start_time: Original start_time of first sentence (to subtract)
        padding: Padding to add to first sentence start (default: 0.3)
    
    Returns:
        Updated JSON part with adjusted timestamps
    """
    updated_part = []
    
    for i, entry in enumerate(json_part):
        updated_entry = entry.copy()
        
        # Update top-level timestamps
        if 'start_time' in updated_entry and updated_entry['start_time'] is not None:
            if i == 0:
                # First sentence: set to padding
                updated_entry['start_time'] = padding
            else:
                # Other sentences: subtract original_start_time
                updated_entry['start_time'] = updated_entry['start_time'] - original_start_time
        
        if 'end_time' in updated_entry and updated_entry['end_time'] is not None:
            updated_entry['end_time'] = updated_entry['end_time'] - original_start_time
        
        # Update matched_words timestamps
        if 'matched_words' in updated_entry and updated_entry['matched_words'] is not None:
            updated_words = []
            for word_idx, word in enumerate(updated_entry['matched_words']):
                updated_word = word.copy()
                if 'start' in updated_word and updated_word['start'] is not None:
                    if i == 0 and word_idx == 0:
                        # First word of first sentence: set to padding
                        updated_word['start'] = padding
                    else:
                        updated_word['start'] = updated_word['start'] - original_start_time
                if 'end' in updated_word and updated_word['end'] is not None:
                    updated_word['end'] = updated_word['end'] - original_start_time
                updated_words.append(updated_word)
            updated_entry['matched_words'] = updated_words
        
        updated_part.append(updated_entry)
    
    return updated_part


def get_first_sentence_from_chapter(chapter_file: Path) -> str:
    """Extract the first sentence from a chapter text file."""
    with open(chapter_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    return first_line


def split_json_and_audio_by_chapters(
    json_dir: Path,
    text_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    padding: float = 0.3
) -> None:
    """
    Main function to split JSON and audio files by chapter boundaries.
    
    Args:
        json_dir: Path to output_thienthan/ directory
        text_dir: Path to data/Text-ThienThanNhoCuaToi/ directory
        audio_dir: Path to data/Audio-ThienThanNhoCuaToi/ directory
        output_dir: Path for output split files
        padding: Time padding in seconds (default: 0.3)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track-to-chapter mapping
    track_configs = {
        'Track_1': {
            'json_file': json_dir / 'Track_1_word_level_matches.json',
            'audio_file': audio_dir / 'Track 1.mp3',
            'splits': [
                {'chapter': 'Chuong2.txt', 'part': 2}
            ]
        },
        'Track_2': {
            'json_file': json_dir / 'Track_2_word_level_matches.json',
            'audio_file': audio_dir / 'Track 2.mp3',
            'splits': [
                {'chapter': 'Chuong4.txt', 'part': 2}
            ]
        },
        'Track_3': {
            'json_file': json_dir / 'Track_3_word_level_matches.json',
            'audio_file': audio_dir / 'Track 3.mp3',
            'splits': [
                {'chapter': 'Chuong6.txt', 'part': 2}
            ]
        },
        'Track_4': {
            'json_file': json_dir / 'Track_4_word_level_matches.json',
            'audio_file': audio_dir / 'Track 4.mp3',
            'splits': [
                {'chapter': 'Chuong7.txt', 'part': 1, 'line_number': 1},
                {'chapter': 'Chuong8.txt', 'part': 2, 'line_number': 185},
                {'chapter': 'Chuong9.txt', 'part': 3, 'line_number': 365}
            ]
        },
        'Track_5': {
            'json_file': json_dir / 'Track_5_word_level_matches.json',
            'audio_file': audio_dir / 'Track 5.mp3',
            'splits': [
                {'chapter': 'Chuong11.txt', 'part': 2}
            ]
        }
    }
    
    for track_name, config in track_configs.items():
        print(f"\n{'='*80}")
        print(f"Processing {track_name}")
        print(f"{'='*80}")
        
        json_file = config['json_file']
        audio_file = config['audio_file']
        
        # Verify files exist
        if not json_file.exists():
            print(f"  Error: JSON file not found: {json_file}")
            continue
        if not audio_file.exists():
            print(f"  Error: Audio file not found: {audio_file}")
            continue
        
        # Load JSON data
        print(f"  Loading JSON: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        print(f"  Loaded {len(json_data)} entries")
        
        # Find split points
        split_points = []
        
        # Add start point (line_number 1, start_time of first entry)
        first_entry = json_data[0]
        first_start_time = first_entry.get('start_time', 0.0)
        split_points.append({
            'line_number': 1,
            'start_time': first_start_time,
            'part': 1
        })
        
        # Find other split points
        for split_config in config['splits']:
            chapter_name = split_config['chapter']
            part_num = split_config['part']
            
            # Check if line_number is explicitly provided (for Track_4)
            if 'line_number' in split_config:
                line_num = split_config['line_number']
                # Find the entry with this line_number
                for entry in json_data:
                    if entry.get('line_number') == line_num:
                        split_points.append({
                            'line_number': line_num,
                            'start_time': entry.get('start_time', 0.0),
                            'part': part_num
                        })
                        print(f"  Found split point at line_number {line_num} (Part {part_num})")
                        break
                else:
                    print(f"  Warning: Could not find line_number {line_num} in JSON")
            else:
                # Find by matching first sentence
                chapter_file = text_dir / chapter_name
                if not chapter_file.exists():
                    print(f"  Error: Chapter file not found: {chapter_file}")
                    continue
                
                target_sentence = get_first_sentence_from_chapter(chapter_file)
                print(f"  Looking for: {target_sentence[:50]}...")
                
                result = find_split_point(json_data, target_sentence)
                if result:
                    line_num, start_time = result
                    split_points.append({
                        'line_number': line_num,
                        'start_time': start_time,
                        'part': part_num
                    })
                    print(f"  Found split point at line_number {line_num}, start_time {start_time:.2f}s (Part {part_num})")
                else:
                    print(f"  Error: Could not find split point for {chapter_name}")
                    continue
        
        # Sort split points by line_number
        split_points.sort(key=lambda x: x['line_number'])
        
        # Add end point
        last_entry = json_data[-1]
        last_end_time = last_entry.get('end_time', 0.0)
        split_points.append({
            'line_number': len(json_data) + 1,  # Beyond last entry
            'start_time': last_end_time,
            'part': len(split_points) + 1
        })
        
        print(f"\n  Split points: {len(split_points) - 1} parts")
        for i, sp in enumerate(split_points[:-1]):
            print(f"    Part {sp['part']}: line_number {sp['line_number']} - {split_points[i+1]['line_number']-1}")
        
        # Process each part
        for part_idx in range(len(split_points) - 1):
            start_split = split_points[part_idx]
            end_split = split_points[part_idx + 1]
            
            part_num = start_split['part']
            start_line = start_split['line_number']
            end_line = end_split['line_number'] - 1
            
            print(f"\n  Processing Part {part_num} (lines {start_line}-{end_line})")
            
            # Extract JSON part
            json_part = []
            for entry in json_data:
                line_num = entry.get('line_number')
                if line_num is not None and start_line <= line_num <= end_line:
                    json_part.append(entry)
            
            if not json_part:
                print(f"    Warning: No entries found for Part {part_num}")
                continue
            
            # Get original timestamps for audio cutting
            first_entry_original = json_part[0]
            last_entry_original = json_part[-1]
            
            # Get start_time from first entry
            original_start_time = first_entry_original.get('start_time')
            if original_start_time is None and first_entry_original.get('matched_words'):
                matched_words = first_entry_original.get('matched_words', [])
                if matched_words and isinstance(matched_words[0], dict) and 'start' in matched_words[0]:
                    original_start_time = matched_words[0]['start']
            if original_start_time is None:
                original_start_time = 0.0
            
            # Get end_time from last entry
            original_end_time = last_entry_original.get('end_time')
            if original_end_time is None and last_entry_original.get('matched_words'):
                matched_words = last_entry_original.get('matched_words', [])
                if matched_words and isinstance(matched_words[-1], dict) and 'end' in matched_words[-1]:
                    original_end_time = matched_words[-1]['end']
            
            # If still no end_time, estimate from previous entry or use a default
            if original_end_time is None:
                # Try to get from previous entry in the part
                if len(json_part) > 1:
                    prev_entry = json_part[-2]
                    prev_end = prev_entry.get('end_time')
                    if prev_end is not None:
                        # Estimate: previous end + 2 seconds
                        original_end_time = prev_end + 2.0
                    elif prev_entry.get('matched_words'):
                        prev_matched = prev_entry.get('matched_words', [])
                        if prev_matched and isinstance(prev_matched[-1], dict) and 'end' in prev_matched[-1]:
                            original_end_time = prev_matched[-1]['end'] + 2.0
                
                # If still None, use start_time + a reasonable duration
                if original_end_time is None:
                    original_end_time = original_start_time + 10.0  # Default 10 seconds
                    print(f"    Warning: Could not determine end_time, using estimate: {original_end_time:.2f}s")
            
            print(f"    Original timestamps: {original_start_time:.2f}s - {original_end_time:.2f}s")
            
            # Cut audio using ORIGINAL timestamps
            output_audio_file = output_dir / f"{track_name}_Part_{part_num}.mp3"
            print(f"    Cutting audio...")
            cut_audio_segment(
                audio_file,
                original_start_time,
                original_end_time,
                output_audio_file,
                padding=0.0  # No padding when cutting, we'll add it in JSON
            )
            
            # Update timestamps in JSON part
            print(f"    Updating timestamps (subtract {original_start_time:.2f}s, add {padding}s padding)...")
            updated_json_part = update_timestamps(json_part, original_start_time, padding)
            
            # Save updated JSON
            output_json_file = output_dir / f"{track_name}_Part_{part_num}_word_level_matches.json"
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(updated_json_part, f, ensure_ascii=False, indent=2)
            
            print(f"    Saved: {output_json_file.name} ({len(updated_json_part)} entries)")
        
        print(f"\n  ✓ Completed {track_name}")
    
    print(f"\n{'='*80}")
    print("All tracks processed!")
    print(f"{'='*80}")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split JSON word-level match files and audio files by chapter boundaries'
    )
    parser.add_argument(
        '--json-dir',
        type=Path,
        default=Path('output_thienthan'),
        help='Path to output_thienthan/ directory (default: output_thienthan)'
    )
    parser.add_argument(
        '--text-dir',
        type=Path,
        default=Path('data/Text-ThienThanNhoCuaToi'),
        help='Path to data/Text-ThienThanNhoCuaToi/ directory (default: data/Text-ThienThanNhoCuaToi)'
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        default=Path('data/Audio-ThienThanNhoCuaToi'),
        help='Path to data/Audio-ThienThanNhoCuaToi/ directory (default: data/Audio-ThienThanNhoCuaToi)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Path for output split files (required)'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.3,
        help='Time padding in seconds (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    split_json_and_audio_by_chapters(
        json_dir=args.json_dir,
        text_dir=args.text_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        padding=args.padding
    )


if __name__ == '__main__':
    main()

