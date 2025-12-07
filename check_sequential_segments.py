"""
Check what's actually in segments 59-120 to see why sentences 24-26 aren't matching.
"""

import json
from match_sentence_to_segments import count_words

# Load segments
with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# The sentences we expect to find
expected_sentences = [
    "- Chào cô giáo Loan!",  # Sentence 24
    "Cô yên tâm đi!",  # Sentence 25
    "Cái bếp dầu tròn đùn khói kia chỉ là vật đối chứng với cái bếp dầu không bấc, không khói, tiết kiệm tối đa nhiên liệu tôi mới mang từ Triển lãm Kỹ thuật - Công nghệ thành phố về, biểu diễn cho bà con biết thôi, hà!",  # Sentence 26
]

print("=" * 80)
print("INVESTIGATING: What's in segments 59-120?")
print("=" * 80)
print("\nSentence 23 ended at segment 58")
print("\nExpected next sentences:")
for i, sent in enumerate(expected_sentences, 24):
    print(f"  {i}. {sent[:70]}...")
print("\n" + "-" * 80)

# Show segments 59-120
print("\nActual segment content (59-120):")
print("-" * 80)

for i in range(59, 121):
    if i >= len(segments):
        break
    seg = segments[i]
    wc = count_words(seg['text'])
    print(f"[{seg['id']:3d}] {seg['start']:6.2f}s-{seg['end']:6.2f}s ({wc:2d}w): {seg['text']}")

print("\n" + "=" * 80)
print("Analysis: Look for patterns matching the expected sentences above")
print("=" * 80)

