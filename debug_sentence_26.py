"""
Debug why sentence 26 and onwards are not matching.
"""

import json
from match_sentence_to_segments import (
    match_sentence_to_segments, 
    count_words,
    normalize_text
)


# Load segments
with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Load text file
with open('data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract sentences (remove line numbers)
sentences = []
for line in lines:
    if '|' in line:
        sentence = line.split('|', 1)[1].strip()
    else:
        sentence = line.strip()
    if sentence:
        sentences.append(sentence)

print("=" * 80)
print("DEBUGGING: Why sentence 26+ are not matching")
print("=" * 80)

# First, let's see what segment we should be at after sentence 25
print("\n1️⃣ Checking sentences 20-25 to find the current segment position...")
print("-" * 80)

current_segment_id = 0
for i in range(20, 26):  # sentences 20-25 (0-indexed: 19-24)
    sentence = sentences[i]
    print(f"\nSentence {i+1}: {sentence[:60]}...")
    
    result = match_sentence_to_segments(
        sentence=sentence,
        segments=segments,
        start_segment_id=current_segment_id,
        max_word_diff=2,
        min_similarity=0.5
    )
    
    if result:
        print(f"  ✓ Matched to segments {result['segment_ids']}")
        print(f"    Last segment ID: {result['segment_ids'][-1]}")
        current_segment_id = result['segment_ids'][-1] + 1
    else:
        print(f"  ✗ NO MATCH")
        print(f"    Current search position: segment {current_segment_id}")

print("\n" + "=" * 80)
print(f"After sentence 25, next search should start at segment: {current_segment_id}")
print("=" * 80)

# Now let's check sentence 26
print("\n2️⃣ Testing sentence 26...")
print("-" * 80)

sentence_26 = sentences[25]  # 0-indexed
print(f"Sentence 26: {sentence_26}")
print(f"Word count: {count_words(sentence_26)}")

# Try matching from current position
print(f"\nAttempting match starting from segment {current_segment_id}...")

result = match_sentence_to_segments(
    sentence=sentence_26,
    segments=segments,
    start_segment_id=current_segment_id,
    max_word_diff=2,
    min_similarity=0.5
)

if result:
    print(f"\n✓ MATCHED!")
    print(f"  Segments: {result['segment_ids']}")
    print(f"  Similarity: {result['similarity']:.2%}")
    print(f"  Words: {result['word_count_sentence']} → {result['word_count_segments']}")
else:
    print(f"\n✗ NO MATCH FOUND")
    print(f"\nLet's check the next few segments manually...")
    print("-" * 80)
    
    # Show next 10 segments
    for i in range(current_segment_id, min(current_segment_id + 10, len(segments))):
        seg = segments[i]
        wc = count_words(seg['text'])
        print(f"[{seg['id']:3d}] {seg['start']:6.2f}s-{seg['end']:6.2f}s ({wc:2d}w): {seg['text'][:70]}")
    
    print("\n" + "-" * 80)
    print("Let's try with lower similarity threshold (0.3)...")
    
    result2 = match_sentence_to_segments(
        sentence=sentence_26,
        segments=segments,
        start_segment_id=current_segment_id,
        max_word_diff=3,
        min_similarity=0.3
    )
    
    if result2:
        print(f"\n✓ MATCHED with lower threshold!")
        print(f"  Segments: {result2['segment_ids']}")
        print(f"  Similarity: {result2['similarity']:.2%}")
        print(f"  Words: {result2['word_count_sentence']} → {result2['word_count_segments']}")
        print(f"\n  Combined text:")
        print(f"  {result2['combined_text'][:200]}...")
    else:
        print("\n✗ Still no match even with lower threshold")
        
        # Try searching from beginning
        print("\nLet's try searching from the beginning (segment 0)...")
        result3 = match_sentence_to_segments(
            sentence=sentence_26,
            segments=segments,
            start_segment_id=0,
            max_word_diff=3,
            min_similarity=0.3
        )
        
        if result3:
            print(f"\n⚠️  FOUND, but at wrong position!")
            print(f"  Segments: {result3['segment_ids']} (expected around {current_segment_id})")
            print(f"  This means segments are out of order or we skipped something")
        else:
            print("\n❌ Cannot find this sentence anywhere in segments")

print("\n" + "=" * 80)

