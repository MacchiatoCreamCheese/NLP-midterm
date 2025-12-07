"""
Test matching the 8th sentence from the text file.
"""

import json
from match_sentence_to_segments import match_sentence_to_segments, count_words


# Load segments
with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# The 8th sentence from the text file
sentence_8 = ("Nóng oi quá, có lẽ vì vậy mà con chó Tép nòi Tây Ban Nha tai cụp, lông trắng "
              "vá nâu không ngủ được, cả đêm sủa nhăng nhẳng, đến mức ông Khái, chủ nhân, một "
              "nhà cách mạng lão thành, ở khu tập thể này mất giấc ngủ trưa như thường lệ, tỉnh "
              "dậy, sau khi quát:")

print("=" * 80)
print("Testing Sentence 8: 'Nóng oi quá...'")
print("=" * 80)
print(f"\nSentence:")
print(f"  {sentence_8}")
print(f"\nWord count: {count_words(sentence_8)} words")

# Previous sentences would have consumed segments 0-8, so start from around segment 9
# But let's start from 0 to find it properly
match_result = match_sentence_to_segments(
    sentence=sentence_8,
    segments=segments,
    start_segment_id=0,  # Start from beginning to find it
    max_word_diff=2,
    min_similarity=0.5
)

if match_result:
    print("\n" + "-" * 80)
    print(f"✓ MATCHED TO {len(match_result['matched_segments'])} SEGMENT(S)!")
    print("-" * 80)
    print(f"Segment IDs: {match_result['segment_ids']}")
    print(f"Time range: {match_result['start_time']:.2f}s - {match_result['end_time']:.2f}s")
    print(f"Duration: {match_result['end_time'] - match_result['start_time']:.2f}s")
    print(f"Similarity score: {match_result['similarity']:.2%}")
    print(f"Word count - Sentence: {match_result['word_count_sentence']}, "
          f"Segments: {match_result['word_count_segments']}")
    print(f"Word difference: {abs(match_result['word_count_sentence'] - match_result['word_count_segments'])}")
    
    print("\n" + "-" * 80)
    print("Individual segments:")
    print("-" * 80)
    for i, seg in enumerate(match_result['matched_segments'], 1):
        word_count = count_words(seg['text'])
        print(f"{i}. [ID:{seg['id']:3d}] {seg['start']:6.2f}s - {seg['end']:6.2f}s ({word_count} words)")
        print(f"   Text: \"{seg['text']}\"")
        print()
    
    print("-" * 80)
    print("Combined text from all segments:")
    print("-" * 80)
    print(f"{match_result['combined_text']}")
    print("\n" + "=" * 80)
    print("✅ TEST PASSED - Match found successfully!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED - No match found")
    print("=" * 80)

