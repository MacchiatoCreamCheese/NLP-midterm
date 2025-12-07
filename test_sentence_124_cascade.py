#!/usr/bin/env python3
"""
Test for cascading error around sentence 124.
Diagnose why matching fails starting from sentence 124.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match_sentence_words import (
    extract_words_from_sentence,
    find_word_sequence,
)

def test_around_sentence_124():
    """Test matching around sentence 124 to find cascading error."""
    
    print("=" * 80)
    print("DIAGNOSING: Why matching fails from sentence 124 onwards")
    print("=" * 80)
    
    text_file = "data/Text-XaXoiThonNguaGia/Cánh bướm tím.txt"
    words_file = "whisper_output_xaxoi/Cánh bướm tím/whisper_words.json"
    
    # Load sentences
    print(f"\nLoading text file: {text_file}")
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
    
    print(f"Total sentences: {len(sentences)}")
    
    # Load words
    print(f"\nLoading whisper words...")
    with open(words_file, 'r', encoding='utf-8') as f:
        all_words = json.load(f)
    
    print(f"Total words: {len(all_words)}")
    
    # Find where sentence 123 matched
    print("\n" + "=" * 80)
    print("Finding where sentence 123 matched...")
    print("=" * 80)
    
    current_word_idx = 0
    
    # Test sentences 120-123 to find the last successful match
    for i in range(119, 124):
        if i >= len(sentences):
            break
        
        sentence = sentences[i]
        sentence_words = extract_words_from_sentence(sentence)
        
        print(f"\n[{i+1}] {sentence[:60]}...")
        print(f"    Search from word {current_word_idx}")
        
        match_result = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=500,
            min_similarity=0.6,
            max_word_gap=5
        )
        
        if match_result:
            print(f"    ✓ MATCHED at words {match_result['start_word_idx']}-{match_result['end_word_idx']}")
            print(f"      Time: {match_result['start_time']:.2f}s - {match_result['end_time']:.2f}s")
            current_word_idx = match_result['end_word_idx'] + 1
        else:
            print(f"    ✗ NO MATCH - Cascade starts here!")
            break
    
    print(f"\nAfter sentence {i+1}, search position: word {current_word_idx}")
    
    # Test sentences 124-130
    print("\n" + "=" * 80)
    print("Testing sentences 124-130...")
    print("=" * 80)
    
    for test_idx in range(123, min(130, len(sentences))):
        sentence = sentences[test_idx]
        sentence_words = extract_words_from_sentence(sentence)
        
        print(f"\n[{test_idx+1}] {sentence[:70]}...")
        print(f"    Words: {' '.join(sentence_words[:8])}...")
        
        # Test 1: Sequential search
        print(f"    Sequential from {current_word_idx}: ", end="")
        match1 = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=500,
            min_similarity=0.6,
            max_word_gap=5
        )
        
        if match1:
            print(f"✓ Found at {match1['start_word_idx']}-{match1['end_word_idx']}")
            current_word_idx = match1['end_word_idx'] + 1
        else:
            print(f"✗")
            
            # Test 2: Search from beginning (see if it exists)
            print(f"    Search from beginning: ", end="")
            match2 = find_word_sequence(
                sentence_words=sentence_words,
                all_words=all_words,
                start_word_idx=0,
                max_search_window=len(all_words),
                min_similarity=0.5,
                max_word_gap=10
            )
            
            if match2:
                print(f"✓ Found at {match2['start_word_idx']}-{match2['end_word_idx']}")
                if match2['start_word_idx'] < current_word_idx:
                    print(f"    ⚠️  CASCADE DETECTED! We passed it at word {match2['start_word_idx']}")
                    print(f"       Current position {current_word_idx} is AFTER where it actually is!")
                    current_word_idx = match2['end_word_idx'] + 1  # Fix position
                else:
                    current_word_idx = match2['end_word_idx'] + 1
            else:
                print(f"✗ Not found anywhere")
                
                # Show context
                print(f"    Context around word {current_word_idx}:")
                start = max(0, current_word_idx - 3)
                end = min(len(all_words), current_word_idx + 15)
                for idx in range(start, end):
                    word = all_words[idx]['word']
                    marker = " <-- HERE" if idx == current_word_idx else ""
                    print(f"      [{idx}] '{word}'{marker}")

if __name__ == "__main__":
    test_around_sentence_124()

