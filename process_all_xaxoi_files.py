#!/usr/bin/env python3
"""
Process all files in the XaXoiThonNguaGia dataset:
1. Run Whisper transcription
2. Match sentences using Method W (word-level matching)
3. Cut audio segments based on word-level matches

This script processes all audio/text pairs in the dataset.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


# Configuration
AUDIO_DIR = "data/Audio-XaXoiThonNguaGia"
TEXT_DIR = "data/Text-XaXoiThonNguaGia"
WHISPER_OUTPUT_DIR = "whisper_output_xaxoi"
OUTPUT_DIR = "output_xaxoi"
AUDIO_SEGMENTS_DIR = "output_xaxoi/audio_segments_method_w"

# Whisper model to use
WHISPER_MODEL = "base"

# Padding for audio cutting (seconds)
AUDIO_PADDING = 0.3


def get_file_pairs():
    """
    Get all audio/text file pairs from the dataset.
    
    Returns:
        List of (audio_file, text_file, base_name) tuples
    """
    audio_dir = Path(AUDIO_DIR)
    text_dir = Path(TEXT_DIR)
    
    pairs = []
    
    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.mp3"))
    
    for audio_file in audio_files:
        base_name = audio_file.stem  # Without extension
        
        # Try to find matching text file
        # Check common name variations
        text_file = None
        possible_names = [
            base_name,
            base_name.replace("XaXoiThonNguaGia", "XaXoiThonNguaGia"),  # Already matches
            base_name.replace("P1", "P1"),
            base_name.replace("P2", "P2"),
            base_name.replace("P3", "P3"),
        ]
        
        for name in possible_names:
            text_candidate = text_dir / f"{name}.txt"
            if text_candidate.exists():
                text_file = text_candidate
                break
        
        if text_file and text_file.exists():
            pairs.append((audio_file, text_file, base_name))
        else:
            print(f"⚠️  Warning: No matching text file found for {audio_file.name}")
    
    return pairs


def run_whisper(audio_file, output_dir, model=WHISPER_MODEL):
    """
    Run Whisper transcription on an audio file.
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save Whisper results
        model: Whisper model to use
    
    Returns:
        Path to whisper_output directory for this file, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Running Whisper on: {audio_file.name}")
    print(f"{'='*80}")
    
    # Create output directory for this file
    base_name = Path(audio_file).stem
    file_output_dir = Path(output_dir) / base_name
    
    print(f"Output directory: {file_output_dir}")
    
    # Check if already exists
    whisper_words = file_output_dir / "whisper_words.json"
    if whisper_words.exists():
        print(f"✓ Whisper results already exist, skipping transcription")
        return file_output_dir
    
    # Import align_vietnamese_audio to use Whisper function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from align_vietnamese_audio import run_whisper_transcription
    
    try:
        print(f"Running Whisper transcription (this may take a while)...")
        run_whisper_transcription(str(audio_file), model, str(file_output_dir))
        print(f"✓ Whisper transcription complete!")
        return file_output_dir
    except Exception as e:
        print(f"❌ Error running Whisper: {e}")
        return None


def run_method_w_matching(text_file, whisper_words_file, output_json):
    """
    Run Method W word-level matching.
    
    Args:
        text_file: Path to text file with sentences
        whisper_words_file: Path to whisper_words.json
        output_json: Path to save word_level_matches.json
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running Method W word-level matching")
    print(f"{'='*80}")
    print(f"Text file:        {text_file}")
    print(f"Whisper words:    {whisper_words_file}")
    print(f"Output:           {output_json}")
    
    # Check if already exists
    if Path(output_json).exists():
        print(f"✓ Word-level matches already exist, skipping")
        return True
    
    # Import the matching function
    sys.path.insert(0, os.path.dirname(__file__))
    from match_sentence_words import match_sentences_using_words, save_results, print_summary
    
    try:
        print(f"\nMatching sentences to words...")
        results = match_sentences_using_words(
            text_file_path=str(text_file),
            words_file_path=str(whisper_words_file),
            min_similarity=0.6,
            max_word_gap=5,
            max_search_window=500,
            strict_sequential=True,  # Use scoring-based sequential matching
            max_word_jump=100,  # Hard limit to reject truly wrong matches
            max_time_jump=30.0  # Hard limit to reject truly wrong matches
        )
        
        # Save results
        os.makedirs(Path(output_json).parent, exist_ok=True)
        save_results(results, str(output_json))
        
        # Print summary
        print_summary(results)
        
        print(f"✓ Method W matching complete!")
        return True
    except Exception as e:
        print(f"❌ Error in Method W matching: {e}")
        import traceback
        traceback.print_exc()
        return False


def cut_audio_segments(word_matches_json, audio_file, output_dir, padding=AUDIO_PADDING):
    """
    Cut audio segments based on word-level matches.
    
    Args:
        word_matches_json: Path to word_level_matches.json
        audio_file: Path to audio file
        output_dir: Directory to save audio segments
        padding: Padding in seconds
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Cutting audio segments")
    print(f"{'='*80}")
    print(f"Word matches:     {word_matches_json}")
    print(f"Audio file:       {audio_file}")
    print(f"Output directory: {output_dir}")
    
    # Check if already exists and has files
    output_path = Path(output_dir)
    if output_path.exists() and len(list(output_path.glob("sentence_*.wav"))) > 0:
        print(f"✓ Audio segments already exist, skipping")
        return True
    
    # Import the cutting function
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Use the cut_audio_from_word_matches script logic
    from cut_audio_from_word_matches import (
        load_word_matches,
        process_matches_for_cutting
    )
    from align_vietnamese_audio import cut_audio_segments
    
    try:
        # Load matches
        matches = load_word_matches(str(word_matches_json))
        
        # Process for cutting
        sentence_timestamps, stats = process_matches_for_cutting(matches)
        
        if len(sentence_timestamps) == 0:
            print(f"❌ No valid segments to cut")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCutting {len(sentence_timestamps)} audio segments...")
        print(f"  Segments from matched sentences: {stats['matched_used']}")
        print(f"  Segments from filled gaps:       {stats['gaps_filled']}")
        print(f"  Segments with default duration:  {stats['default_duration_used']}")
        
        # Cut audio
        cut_audio_segments(
            str(audio_file),
            sentence_timestamps,
            str(output_dir),
            add_padding=padding
        )
        
        print(f"✓ Audio cutting complete!")
        return True
    except Exception as e:
        print(f"❌ Error cutting audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_file_pair(audio_file, text_file, base_name):
    """
    Process a single audio/text file pair through the entire pipeline.
    
    Args:
        audio_file: Path to audio file
        text_file: Path to text file
        base_name: Base name for output files
    
    Returns:
        True if all steps succeeded, False otherwise
    """
    print(f"\n{'#'*80}")
    print(f"Processing: {base_name}")
    print(f"{'#'*80}")
    print(f"Audio: {audio_file}")
    print(f"Text:  {text_file}")
    
    # Step 1: Run Whisper
    whisper_dir = Path(WHISPER_OUTPUT_DIR) / base_name
    whisper_result = run_whisper(audio_file, WHISPER_OUTPUT_DIR, WHISPER_MODEL)
    
    if not whisper_result:
        print(f"❌ Failed: Whisper transcription failed")
        return False
    
    whisper_words_file = whisper_result / "whisper_words.json"
    if not whisper_words_file.exists():
        print(f"❌ Failed: whisper_words.json not found at {whisper_words_file}")
        return False
    
    # Step 2: Method W matching
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    word_matches_json = Path(OUTPUT_DIR) / f"{base_name}_word_level_matches.json"
    
    if not run_method_w_matching(text_file, whisper_words_file, word_matches_json):
        print(f"❌ Failed: Method W matching failed")
        return False
    
    if not word_matches_json.exists():
        print(f"❌ Failed: word_level_matches.json not created")
        return False
    
    # Step 3: Cut audio
    audio_segments_dir = Path(AUDIO_SEGMENTS_DIR) / base_name
    os.makedirs(audio_segments_dir, exist_ok=True)
    
    if not cut_audio_segments(word_matches_json, audio_file, audio_segments_dir, AUDIO_PADDING):
        print(f"❌ Failed: Audio cutting failed")
        return False
    
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS: {base_name} - All steps completed!")
    print(f"{'='*80}")
    
    return True


def main():
    """Main function to process all files."""
    print("=" * 80)
    print("XaXoiThonNguaGia Dataset - Complete Processing Pipeline")
    print("Method W: Word-Level Matching & Audio Cutting")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Audio directory:      {AUDIO_DIR}")
    print(f"  Text directory:       {TEXT_DIR}")
    print(f"  Whisper output:       {WHISPER_OUTPUT_DIR}")
    print(f"  Method W output:      {OUTPUT_DIR}")
    print(f"  Audio segments:       {AUDIO_SEGMENTS_DIR}")
    print(f"  Whisper model:        {WHISPER_MODEL}")
    print(f"  Audio padding:        {AUDIO_PADDING}s")
    print()
    
    # Get all file pairs
    pairs = get_file_pairs()
    
    if not pairs:
        print("❌ No matching audio/text file pairs found!")
        print(f"   Check that audio files exist in: {AUDIO_DIR}")
        print(f"   And text files exist in: {TEXT_DIR}")
        sys.exit(1)
    
    print(f"Found {len(pairs)} file pairs to process:")
    for audio_file, text_file, base_name in pairs:
        print(f"  - {base_name}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with processing all files? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    print()
    
    # Process each pair
    results = []
    for audio_file, text_file, base_name in pairs:
        success = process_file_pair(audio_file, text_file, base_name)
        results.append((base_name, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    successful = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"   - {name}")
    
    print("\n" + "=" * 80)
    
    if failed:
        sys.exit(1)
    else:
        print("🎉 All files processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

