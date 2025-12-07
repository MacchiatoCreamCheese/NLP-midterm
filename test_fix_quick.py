#!/usr/bin/env python3
"""
Quick test: Run matching on sentences 120-135 to see if fix works.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match_sentence_words import (
    extract_words_from_sentence,
    find_word_sequence,
)

def quick_test():
    """Test sentences 120-135 with the fix."""
    
    text_file = "data/Text-XaXoiThonNguaGia/Cánh bướm tím.txt"
    words_file = "whisper_output_xaxoi/Cánh bướm tím/whisper_words.json"
    
    print("Loading files...")
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    for line in lines:
        if '|' in line:
            sentence = line.split('|', 1)[1].strip()
        else:
            sentence = line.strip()
        if sentence:
            sentences.append(sentence)
    
    with open(words_file, 'r', encoding='utf-8') as f:
        all_words = json.load(f)
    
    print(f"Testing sentences 120-135 with FIXED code...")
    print("=" * 80)
    
    # Find where sentence 119 ended (approximate)
    # Let's start from a reasonable position - we'll test around word 4000-5000
    # Actually, let's start from where sentence 119 might be
    current_word_idx = 0
    
    # First, quickly find where sentence 119 might be
    # We'll test from sentence 115 to get a baseline
    print("\nFinding baseline position from sentence 115...")
    for i in range(114, 120):  # Sentences 115-119
        sentence = sentences[i]
        sentence_words = extract_words_from_sentence(sentence)
        
        match = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=500,
            min_similarity=0.6,
            max_word_gap=5
        )
        
        if match:
            current_word_idx = match['end_word_idx'] + 1
            if i >= 118:  # Show last few
                print(f"Sentence {i+1}: ✓ at words {match['start_word_idx']}-{match['end_word_idx']}")
    
    print(f"\nStarting position for test: word {current_word_idx}")
    print("\n" + "=" * 80)
    print("Testing sentences 120-135 with FIX (conservative skip)...")
    print("=" * 80)
    
    matched_count = 0
    unmatched_count = 0
    
    for i in range(119, min(135, len(sentences))):  # Sentences 120-135
        sentence = sentences[i]
        sentence_words = extract_words_from_sentence(sentence)
        
        print(f"\n[{i+1}] {sentence[:60]}...")
        
        # Try normal match
        match = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=500,
            min_similarity=0.6,
            max_word_gap=5
        )
        
        if match:
            print(f"    ✓ Matched at words {match['start_word_idx']}-{match['end_word_idx']}")
            current_word_idx = match['end_word_idx'] + 1
            matched_count += 1
        else:
            # Try relaxed search (as in the fix)
            print(f"    ✗ Normal match failed, trying relaxed...")
            relaxed = find_word_sequence(
                sentence_words=sentence_words,
                all_words=all_words,
                start_word_idx=max(0, current_word_idx - 50),
                max_search_window=min(500, len(all_words) - max(0, current_word_idx - 50)),
                min_similarity=0.45,
                max_word_gap=8
            )
            
            if relaxed:
                print(f"    ✓ Found with relaxed search at words {relaxed['start_word_idx']}-{relaxed['end_word_idx']}")
                current_word_idx = relaxed['end_word_idx'] + 1
                matched_count += 1
            else:
                print(f"    ✗ Still no match - advancing by 1 word (fix)")
                if current_word_idx < len(all_words) - 1:
                    current_word_idx += 1
                unmatched_count += 1
    
    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"  Matched:   {matched_count}/16")
    print(f"  Unmatched: {unmatched_count}/16")
    print(f"  Success rate: {matched_count/16*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    quick_test()

