#!/usr/bin/env python3
"""
Find where the cascade error actually starts by checking backwards from sentence 124.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match_sentence_words import (
    extract_words_from_sentence,
    find_word_sequence,
)

def find_cascade_start():
    """Find where matching first failed."""
    
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
    
    print(f"Sentences: {len(sentences)}, Words: {len(all_words)}")
    print("\n" + "=" * 80)
    print("Finding last successful match before sentence 124...")
    print("=" * 80)
    
    # Work backwards from sentence 120 to find last good match
    last_good_idx = None
    last_good_word_idx = None
    
    current_word_idx = 0
    
    for i in range(119):  # Check sentences 1-119
        sentence = sentences[i]
        sentence_words = extract_words_from_sentence(sentence)
        
        match_result = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=500,
            min_similarity=0.6,
            max_word_gap=5
        )
        
        if match_result:
            last_good_idx = i
            last_good_word_idx = match_result['end_word_idx'] + 1
            current_word_idx = match_result['end_word_idx'] + 1
            
            if (i + 1) % 20 == 0:
                print(f"Sentence {i+1}: ✓ matched, position at word {current_word_idx}")
        else:
            print(f"\n⚠️  Sentence {i+1} FAILED to match!")
            print(f"   Last good match was sentence {last_good_idx + 1 if last_good_idx is not None else 'NONE'}")
            print(f"   Last good position was word {last_good_word_idx if last_good_word_idx is not None else 'NONE'}")
            break
    
    print(f"\n{'='*80}")
    print(f"Last successful match: Sentence {last_good_idx + 1 if last_good_idx is not None else 'NONE'}")
    print(f"Last good word position: {last_good_word_idx if last_good_word_idx is not None else 'NONE'}")
    
    if last_good_word_idx:
        print(f"\nNow testing sentences 120-125 from position {last_good_word_idx}...")
        current_word_idx = last_good_word_idx
        
        for i in range(119, min(125, len(sentences))):
            sentence = sentences[i]
            sentence_words = extract_words_from_sentence(sentence)
            
            print(f"\n[{i+1}] {sentence[:60]}...")
            
            match_result = find_word_sequence(
                sentence_words=sentence_words,
                all_words=all_words,
                start_word_idx=current_word_idx,
                max_search_window=500,
                min_similarity=0.6,
                max_word_gap=5
            )
            
            if match_result:
                print(f"    ✓ Matched at words {match_result['start_word_idx']}-{match_result['end_word_idx']}")
                current_word_idx = match_result['end_word_idx'] + 1
            else:
                print(f"    ✗ No match from position {current_word_idx}")

if __name__ == "__main__":
    find_cascade_start()

