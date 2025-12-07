"""
Simple example demonstrating how to use the sentence-to-segments matching function.
"""

import json
from match_sentence_to_segments import match_sentence_to_segments, normalize_text, count_words


def demo_single_sentence_match():
    """
    Demonstrate matching a single sentence to multiple segments.
    """
    # Load segments
    with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # Example sentence from the text file (line 3)
    example_sentence = "Tháng mười một Tây rồi mà trời còn nồng nực như giữa mùa hè."
    
    print("=" * 80)
    print("EXAMPLE: Matching a single sentence to multiple segments")
    print("=" * 80)
    print(f"\nSentence to match:")
    print(f"  '{example_sentence}'")
    print(f"  Word count: {count_words(example_sentence)}")
    print(f"  Normalized: '{normalize_text(example_sentence)}'")
    
    # Find matching segments
    match_result = match_sentence_to_segments(
        sentence=example_sentence,
        segments=segments,
        start_segment_id=0,
        max_word_diff=2,
        min_similarity=0.5
    )
    
    if match_result:
        print("\n" + "-" * 80)
        print("MATCH FOUND!")
        print("-" * 80)
        print(f"Number of segments matched: {len(match_result['matched_segments'])}")
        print(f"Segment IDs: {match_result['segment_ids']}")
        print(f"Time range: {match_result['start_time']:.2f}s - {match_result['end_time']:.2f}s")
        print(f"Similarity score: {match_result['similarity']:.2%}")
        print(f"Word count - Sentence: {match_result['word_count_sentence']}, "
              f"Segments: {match_result['word_count_segments']}")
        
        print("\nMatched segments:")
        for seg in match_result['matched_segments']:
            print(f"  [{seg['id']}] {seg['start']:.2f}s-{seg['end']:.2f}s: \"{seg['text']}\"")
        
        print(f"\nCombined text from segments:")
        print(f"  '{match_result['combined_text']}'")
    else:
        print("\n✗ No match found")
    
    print("\n" + "=" * 80)


def demo_multiple_sentences():
    """
    Demonstrate matching first few sentences sequentially.
    """
    # Load segments
    with open('whisper_output/whisper_segments.json', 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # First few sentences from the text
    sentences = [
        "Xa xôi Thôn Ngựa Già.",
        "Phần 1.",
        "Tháng mười một Tây rồi mà trời còn nồng nực như giữa mùa hè.",
        "Quạt máy quay vù vù suốt hai mươi tư giờ không ngừng nghỉ.",
    ]
    
    print("=" * 80)
    print("EXAMPLE: Matching multiple sentences sequentially")
    print("=" * 80)
    
    current_segment_id = 0
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}] Sentence: '{sentence}'")
        
        match_result = match_sentence_to_segments(
            sentence=sentence,
            segments=segments,
            start_segment_id=current_segment_id,
            max_word_diff=2,
            min_similarity=0.5
        )
        
        if match_result:
            print(f"    ✓ Matched to segments: {match_result['segment_ids']}")
            print(f"    Time: {match_result['start_time']:.2f}s - {match_result['end_time']:.2f}s")
            print(f"    Similarity: {match_result['similarity']:.2%}")
            print(f"    Combined: '{match_result['combined_text'][:60]}...'")
            
            # Update for next search
            current_segment_id = match_result['segment_ids'][-1] + 1
        else:
            print(f"    ✗ No match found")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demo_single_sentence_match()
    print("\n\n")
    demo_multiple_sentences()

