"""
Flexible sentence matching that allows for gaps and reordering.
"""

import json
from match_sentence_to_segments import (
    match_sentence_to_segments,
    count_words
)


def match_sentences_flexible(
    text_file_path: str,
    segments_file_path: str,
    max_word_diff: int = 2,
    min_similarity: float = 0.5,
    allow_reordering: bool = True
) -> list:
    """
    Match sentences with flexible search - allows for content gaps and reordering.
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
    used_segments = set()  # Track used segments to avoid duplicates
    current_search_start = 0
    
    for idx, sentence in enumerate(sentences):
        print(f"Matching sentence {idx + 1}/{len(sentences)}: {sentence[:50]}...")
        
        # First try: search from current position with limited window
        match_result = match_sentence_to_segments(
            sentence=sentence,
            segments=segments,
            start_segment_id=current_search_start,
            max_word_diff=max_word_diff,
            min_similarity=min_similarity,
            max_search_window=100  # Look ahead 100 segments
        )
        
        # If no match and reordering allowed, search from beginning (avoiding used segments)
        if not match_result and allow_reordering:
            match_result = match_sentence_to_segments(
                sentence=sentence,
                segments=segments,
                start_segment_id=0,
                max_word_diff=max_word_diff,
                min_similarity=min_similarity - 0.1,  # Lower threshold
                max_search_window=len(segments)  # Search all
            )
            
            if match_result:
                print(f"  ⚠️  Found out of order at segments {match_result['segment_ids']}")
        
        if match_result:
            # Check if segments are already used
            seg_ids = set(match_result['segment_ids'])
            if seg_ids & used_segments:
                print(f"  ⚠️  Warning: Some segments already used!")
            
            used_segments.update(seg_ids)
            
            results.append({
                'line_number': idx + 1,
                **match_result
            })
            
            # Update search position (but allow some flexibility)
            if match_result['segment_ids'][-1] >= current_search_start:
                current_search_start = match_result['segment_ids'][-1] + 1
            
            print(f"  ✓ Matched to segments {match_result['segment_ids']} "
                  f"(similarity: {match_result['similarity']:.2f})")
        else:
            results.append({
                'line_number': idx + 1,
                'sentence': sentence,
                'matched_segments': None,
                'error': 'No match found'
            })
            print(f"  ✗ No match found")
    
    return results


if __name__ == "__main__":
    text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
    segments_file = "whisper_output/whisper_segments.json"
    output_file = "output/flexible_sentence_matches.json"
    
    print("Starting FLEXIBLE sentence-to-segment matching...")
    print(f"Text file: {text_file}")
    print(f"Segments file: {segments_file}")
    print("-" * 80)
    
    results = match_sentences_flexible(
        text_file_path=text_file,
        segments_file_path=segments_file,
        max_word_diff=2,
        min_similarity=0.5,
        allow_reordering=True
    )
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Summary
    successful = sum(1 for r in results if r.get('matched_segments') is not None)
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Total sentences: {len(results)}")
    print(f"  Successfully matched: {successful}")
    print(f"  Failed to match: {len(results) - successful}")
    print(f"  Success rate: {successful/len(results)*100:.1f}%")
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

