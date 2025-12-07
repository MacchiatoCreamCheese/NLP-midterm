"""
Improved matching that can handle cases where multiple text sentences 
are combined into one or more audio segments.
"""

import json
from match_sentence_to_segments import (
    match_sentence_to_segments,
    count_words,
    normalize_text,
    calculate_similarity
)


def match_sentences_with_merging(
    text_file_path: str,
    segments_file_path: str,
    max_word_diff: int = 2,
    min_similarity: float = 0.5,
    max_search_window: int = 50,
    max_sentence_merge: int = 3
) -> list:
    """
    Match sentences, trying to merge multiple sentences if single match fails.
    
    Args:
        max_sentence_merge: Maximum number of consecutive sentences to merge (default: 3)
    """
    # Load segments
    with open(segments_file_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # Load text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract sentences
    sentences = []
    for line in lines:
        if '|' in line:
            sentence = line.split('|', 1)[1].strip()
        else:
            sentence = line.strip()
        if sentence:
            sentences.append(sentence)
    
    results = []
    current_segment_id = 0
    i = 0
    
    while i < len(sentences):
        print(f"Matching sentence {i + 1}/{len(sentences)}: {sentences[i][:50]}...")
        
        matched = False
        
        # Try matching with increasing number of merged sentences
        for merge_count in range(1, min(max_sentence_merge + 1, len(sentences) - i + 1)):
            # Combine sentences
            if merge_count == 1:
                combined_sentence = sentences[i]
                sentence_range = f"{i+1}"
            else:
                combined_sentence = " ".join(sentences[i:i+merge_count])
                sentence_range = f"{i+1}-{i+merge_count}"
            
            # Try to match
            match_result = match_sentence_to_segments(
                sentence=combined_sentence,
                segments=segments,
                start_segment_id=current_segment_id,
                max_word_diff=max_word_diff * merge_count,  # Allow more difference for merged
                min_similarity=min_similarity,
                max_search_window=max_search_window
            )
            
            if match_result:
                if merge_count > 1:
                    print(f"  ✓ Matched by merging sentences {sentence_range} "
                          f"to segments {match_result['segment_ids']} "
                          f"(similarity: {match_result['similarity']:.2f})")
                else:
                    print(f"  ✓ Matched to segments {match_result['segment_ids']} "
                          f"(similarity: {match_result['similarity']:.2f})")
                
                # Add results for each sentence in the merge
                for j in range(merge_count):
                    results.append({
                        'line_number': i + j + 1,
                        'sentence': sentences[i + j],
                        'matched_segments': match_result['matched_segments'],
                        'combined_text': match_result['combined_text'],
                        'word_count_sentence': count_words(sentences[i + j]),
                        'word_count_segments': match_result['word_count_segments'] if j == 0 else None,
                        'similarity': match_result['similarity'],
                        'start_time': match_result['start_time'],
                        'end_time': match_result['end_time'],
                        'segment_ids': match_result['segment_ids'],
                        'merged_with': merge_count if merge_count > 1 else None,
                        'merge_group': sentence_range if merge_count > 1 else None
                    })
                
                # Update position
                current_segment_id = match_result['segment_ids'][-1] + 1
                i += merge_count
                matched = True
                break
        
        if not matched:
            # No match found even with merging
            results.append({
                'line_number': i + 1,
                'sentence': sentences[i],
                'matched_segments': None,
                'error': 'No match found'
            })
            print(f"  ✗ No match found")
            i += 1
    
    return results


if __name__ == "__main__":
    text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
    segments_file = "whisper_output/whisper_segments.json"
    output_file = "output/merged_sentence_matches.json"
    
    print("🔄 Starting sentence matching WITH MERGING support...")
    print(f"Text file: {text_file}")
    print(f"Segments file: {segments_file}")
    print("-" * 80)
    
    results = match_sentences_with_merging(
        text_file_path=text_file,
        segments_file_path=segments_file,
        max_word_diff=2,
        min_similarity=0.5,
        max_search_window=50,
        max_sentence_merge=3  # Try merging up to 3 sentences
    )
    
    # Save results
    print("\n💾 Saving results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Summary
    successful = sum(1 for r in results if r.get('matched_segments') is not None)
    merged_matches = sum(1 for r in results if r.get('merged_with'))
    
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Total sentences: {len(results)}")
    print(f"  Successfully matched: {successful}")
    print(f"  Failed to match: {len(results) - successful}")
    print(f"  Merged matches: {merged_matches}")
    print(f"  Success rate: {successful/len(results)*100:.1f}%")
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

