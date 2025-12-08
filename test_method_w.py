"""
Quick test of Method W (Word-level matching) on first 10 sentences.
"""

import json
from match_sentence_words import (
    match_sentences_using_words,
    print_summary
)

# Test with first 10 sentences only
text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
words_file = "whisper_output/whisper_words.json"

print("=" * 80)
print("METHOD W TEST: Word-Level Matching (First 30 sentences)")
print("=" * 80)
print()

# Load and get first 10 sentences
with open(text_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

sentences = []
for line in lines[:30]:  # First 30 lines
    if '|' in line:
        sentence = line.split('|', 1)[1].strip()
    else:
        sentence = line.strip()
    if sentence:
        sentences.append(sentence)

# Load all words
with open(words_file, 'r', encoding='utf-8') as f:
    all_words = json.load(f)

print(f"Testing with {len(sentences)} sentences")
print(f"Word database has {len(all_words)} words")
print("-" * 80)
print()

# Create temporary test file
test_file = "temp_test_sentences.txt"
with open(test_file, 'w', encoding='utf-8') as f:
    for i, sent in enumerate(sentences, 1):
        f.write(f"{i}|{sent}\n")

# Run matching
results = match_sentences_using_words(
    text_file_path=test_file,
    words_file_path=words_file,
    min_similarity=0.6,
    max_word_gap=5,
    max_search_window=500,
    strict_sequential=True,  # Use scoring-based sequential matching
    max_word_jump=100,  # Hard limit to reject truly wrong matches
    max_time_jump=30.0  # Hard limit to reject truly wrong matches
)

# Print summary
print_summary(results)

# Show first few matches in detail
print("\n" + "=" * 80)
print("DETAILED VIEW: First 5 matches")
print("=" * 80)

for i, result in enumerate(results[:5]):
    if result.get('matched_words'):
        print(f"\n{i+1}. Sentence: {result['sentence'][:70]}...")
        print(f"   Words: {' '.join(result['sentence_words'])}")
        print(f"   Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
        print(f"   Matched: {result['matched_text'][:70]}...")
        print(f"   Similarity: {result['similarity']:.2%}")

# Clean up
import os
os.remove(test_file)

print("\n" + "=" * 80)
print("✅ Method W test complete!")
print("=" * 80)

