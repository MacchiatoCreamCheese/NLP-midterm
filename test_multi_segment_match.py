"""
Test matching a long sentence to multiple segments.
"""

import json
from match_sentence_to_segments import match_sentence_to_segments, count_words


# Load segments
with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Test with a very long sentence (line 5 from the text file)
long_sentence = ("Trừ khi phải ra ngoài, còn suốt ngày ở trong nhà, đánh cái quần đùi "
                "với chiếc may ô cũ, đến bữa khoanh tròn trên nền đá hoa, chan nước canh "
                "dầm sấu húp sùm sụp mà người vẫn cứ là nẫu nà bã bượi, bải hoải như không "
                "còn gân cốt.")

print("=" * 80)
print("Testing a LONG sentence that should match MULTIPLE segments")
print("=" * 80)
print(f"\nSentence:")
print(f"  '{long_sentence}'")
print(f"\nWord count: {count_words(long_sentence)} words")

# Find matching segments starting from segment 4
match_result = match_sentence_to_segments(
    sentence=long_sentence,
    segments=segments,
    start_segment_id=4,
    max_word_diff=2,
    min_similarity=0.5
)

if match_result:
    print("\n" + "-" * 80)
    print(f"✓ MATCHED TO {len(match_result['matched_segments'])} SEGMENTS!")
    print("-" * 80)
    print(f"Segment IDs: {match_result['segment_ids']}")
    print(f"Time range: {match_result['start_time']:.2f}s - {match_result['end_time']:.2f}s")
    print(f"Duration: {match_result['end_time'] - match_result['start_time']:.2f}s")
    print(f"Similarity score: {match_result['similarity']:.2%}")
    print(f"Word count - Sentence: {match_result['word_count_sentence']}, "
          f"Segments: {match_result['word_count_segments']}")
    
    print("\n" + "-" * 80)
    print("Individual segments:")
    print("-" * 80)
    for i, seg in enumerate(match_result['matched_segments'], 1):
        print(f"{i}. [ID:{seg['id']:3d}] {seg['start']:6.2f}s - {seg['end']:6.2f}s")
        print(f"   Text: \"{seg['text']}\"")
        print()
    
    print("-" * 80)
    print("Combined text from all segments:")
    print("-" * 80)
    print(f"'{match_result['combined_text']}'")
    print()
else:
    print("\n✗ No match found")

print("=" * 80)

