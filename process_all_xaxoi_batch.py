#!/usr/bin/env python3
"""
Batch version - processes all files without user interaction.
Use this for automated processing or if you want to process all files automatically.

Usage:
    python process_all_xaxoi_batch.py [--skip-whisper] [--skip-matching] [--skip-cutting]
"""

import os
import sys
import argparse

# Import main processing function
from process_all_xaxoi_files import (
    get_file_pairs,
    run_whisper,
    run_method_w_matching,
    cut_audio_segments,
    WHISPER_OUTPUT_DIR,
    OUTPUT_DIR,
    AUDIO_SEGMENTS_DIR,
    WHISPER_MODEL,
    AUDIO_PADDING
)
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all XaXoiThonNguaGia files with Method W"
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper transcription step"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip Method W matching step"
    )
    parser.add_argument(
        "--skip-cutting",
        action="store_true",
        help="Skip audio cutting step"
    )
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"Whisper model to use (default: {WHISPER_MODEL})"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=AUDIO_PADDING,
        help=f"Audio padding in seconds (default: {AUDIO_PADDING})"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("XaXoiThonNguaGia Dataset - Batch Processing")
    print("Method W: Word-Level Matching & Audio Cutting")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Whisper model:        {args.model}")
    print(f"  Audio padding:        {args.padding}s")
    print(f"  Skip Whisper:         {args.skip_whisper}")
    print(f"  Skip Matching:        {args.skip_matching}")
    print(f"  Skip Cutting:         {args.skip_cutting}")
    print()
    
    # Get all file pairs
    pairs = get_file_pairs()
    
    if not pairs:
        print("❌ No matching audio/text file pairs found!")
        sys.exit(1)
    
    print(f"Found {len(pairs)} file pairs to process:\n")
    for audio_file, text_file, base_name in pairs:
        print(f"  - {base_name}")
    print()
    
    # Process each pair
    results = []
    for i, (audio_file, text_file, base_name) in enumerate(pairs, 1):
        print(f"\n{'#'*80}")
        print(f"Processing file {i}/{len(pairs)}: {base_name}")
        print(f"{'#'*80}")
        
        success = True
        
        # Step 1: Whisper
        if not args.skip_whisper:
            whisper_result = run_whisper(audio_file, WHISPER_OUTPUT_DIR, args.model)
            if not whisper_result:
                print(f"❌ Failed: Whisper transcription")
                results.append((base_name, False))
                continue
            whisper_words_file = whisper_result / "whisper_words.json"
            if not whisper_words_file.exists():
                print(f"❌ Failed: whisper_words.json not found")
                results.append((base_name, False))
                continue
        else:
            whisper_words_file = Path(WHISPER_OUTPUT_DIR) / base_name / "whisper_words.json"
            if not whisper_words_file.exists():
                print(f"❌ Failed: whisper_words.json not found (use --skip-whisper only if already done)")
                results.append((base_name, False))
                continue
        
        # Step 2: Method W matching
        if not args.skip_matching:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            word_matches_json = Path(OUTPUT_DIR) / f"{base_name}_word_level_matches.json"
            
            if not run_method_w_matching(text_file, whisper_words_file, word_matches_json):
                print(f"❌ Failed: Method W matching")
                results.append((base_name, False))
                continue
        else:
            word_matches_json = Path(OUTPUT_DIR) / f"{base_name}_word_level_matches.json"
            if not word_matches_json.exists():
                print(f"❌ Failed: word_level_matches.json not found (use --skip-matching only if already done)")
                results.append((base_name, False))
                continue
        
        # Step 3: Cut audio
        if not args.skip_cutting:
            audio_segments_dir = Path(AUDIO_SEGMENTS_DIR) / base_name
            os.makedirs(audio_segments_dir, exist_ok=True)
            
            if not cut_audio_segments(word_matches_json, audio_file, audio_segments_dir, args.padding):
                print(f"❌ Failed: Audio cutting")
                results.append((base_name, False))
                continue
        
        results.append((base_name, True))
        print(f"\n✅ {base_name} - Complete!")
    
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
        sys.exit(1)
    
    print("\n🎉 All files processed successfully!")


if __name__ == "__main__":
    main()

