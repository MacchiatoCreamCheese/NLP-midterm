"""
Search for where sentence 26 actually appears in the segments.
"""

import json
from match_sentence_to_segments import normalize_text, calculate_similarity

# Load segments
with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Sentence 26
sentence_26 = ("Cái bếp dầu tròn đùn khói kia chỉ là vật đối chứng với cái bếp dầu không bấc, "
               "không khói, tiết kiệm tối đa nhiên liệu tôi mới mang từ Triển lãm Kỹ thuật - "
               "Công nghệ thành phố về, biểu diễn cho bà con biết thôi, hà!")

print("Searching for sentence 26 in segments...")
print(f"Sentence: {sentence_26[:80]}...")
print()

normalized_sentence = normalize_text(sentence_26)

# Search through segments for high similarity matches
print("Top 10 most similar segment combinations:")
print("-" * 80)

best_matches = []

for start_idx in range(len(segments)):
    for num_segs in range(1, 15):  # Try combining up to 15 segments
        if start_idx + num_segs > len(segments):
            break
        
        combined_text = " ".join([segments[i]['text'] for i in range(start_idx, start_idx + num_segs)])
        similarity = calculate_similarity(sentence_26, combined_text)
        
        if similarity > 0.4:  # Only keep decent matches
            best_matches.append({
                'start_seg': start_idx,
                'end_seg': start_idx + num_segs - 1,
                'num_segs': num_segs,
                'similarity': similarity,
                'text': combined_text[:100]
            })

# Sort by similarity
best_matches.sort(key=lambda x: x['similarity'], reverse=True)

for i, match in enumerate(best_matches[:10], 1):
    print(f"{i}. Segments [{match['start_seg']}-{match['end_seg']}] "
          f"({match['num_segs']} segs) - Similarity: {match['similarity']:.2%}")
    print(f"   {match['text']}...")
    print()

print("=" * 80)
if best_matches:
    best = best_matches[0]
    print(f"BEST MATCH: Segments [{best['start_seg']}-{best['end_seg']}]")
    print(f"This means sentence 26 should match starting around segment {best['start_seg']}")

