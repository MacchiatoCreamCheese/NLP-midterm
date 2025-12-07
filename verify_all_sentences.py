"""
Verify that all sentences from the text file can be matched to Whisper segments.
This script checks for completeness and shows detailed statistics.
"""

import json
from match_sentence_to_segments import (
    match_sentences_to_segments,
    count_words,
    normalize_text
)


def verify_matching_completeness(results):
    """
    Verify that all sentences were matched and provide detailed statistics.
    
    Args:
        results: List of matching results from match_sentences_to_segments()
    
    Returns:
        Dictionary with verification statistics
    """
    total_sentences = len(results)
    matched_sentences = [r for r in results if r.get('matched_segments') is not None]
    unmatched_sentences = [r for r in results if r.get('matched_segments') is None]
    
    # Calculate statistics
    total_matched = len(matched_sentences)
    total_unmatched = len(unmatched_sentences)
    success_rate = (total_matched / total_sentences * 100) if total_sentences > 0 else 0
    
    # Word count statistics
    sentence_word_counts = [r.get('word_count_sentence', 0) for r in matched_sentences]
    segment_word_counts = [r.get('word_count_segments', 0) for r in matched_sentences]
    word_differences = [abs(r.get('word_count_sentence', 0) - r.get('word_count_segments', 0)) 
                       for r in matched_sentences]
    
    # Similarity statistics
    similarities = [r.get('similarity', 0) for r in matched_sentences]
    
    # Segment usage statistics
    all_segment_ids = []
    for r in matched_sentences:
        if r.get('segment_ids'):
            all_segment_ids.extend(r['segment_ids'])
    
    unique_segments_used = len(set(all_segment_ids))
    total_segments_used = len(all_segment_ids)
    
    # Multi-segment matches
    multi_segment_matches = [r for r in matched_sentences 
                            if len(r.get('matched_segments', [])) > 1]
    
    stats = {
        'total_sentences': total_sentences,
        'total_matched': total_matched,
        'total_unmatched': total_unmatched,
        'success_rate': success_rate,
        'is_complete': total_unmatched == 0,
        'sentence_word_counts': {
            'total': sum(sentence_word_counts),
            'min': min(sentence_word_counts) if sentence_word_counts else 0,
            'max': max(sentence_word_counts) if sentence_word_counts else 0,
            'avg': sum(sentence_word_counts) / len(sentence_word_counts) if sentence_word_counts else 0
        },
        'segment_word_counts': {
            'total': sum(segment_word_counts),
            'min': min(segment_word_counts) if segment_word_counts else 0,
            'max': max(segment_word_counts) if segment_word_counts else 0,
            'avg': sum(segment_word_counts) / len(segment_word_counts) if segment_word_counts else 0
        },
        'word_differences': {
            'max': max(word_differences) if word_differences else 0,
            'avg': sum(word_differences) / len(word_differences) if word_differences else 0,
            'exact_matches': sum(1 for d in word_differences if d == 0)
        },
        'similarities': {
            'min': min(similarities) if similarities else 0,
            'max': max(similarities) if similarities else 0,
            'avg': sum(similarities) / len(similarities) if similarities else 0
        },
        'segments_usage': {
            'unique_segments_used': unique_segments_used,
            'total_segments_used': total_segments_used,
            'avg_segments_per_sentence': total_segments_used / total_matched if total_matched > 0 else 0
        },
        'multi_segment_matches': len(multi_segment_matches),
        'unmatched_sentences': unmatched_sentences
    }
    
    return stats


def print_verification_report(stats, results):
    """
    Print a detailed verification report.
    """
    print("=" * 80)
    print("VERIFICATION REPORT: Sentence-to-Segment Matching Completeness")
    print("=" * 80)
    
    # Overall status
    print("\n📊 OVERALL STATUS")
    print("-" * 80)
    if stats['is_complete']:
        print("✅ ALL SENTENCES MATCHED SUCCESSFULLY!")
    else:
        print(f"⚠️  WARNING: {stats['total_unmatched']} sentence(s) could not be matched")
    
    print(f"\nTotal sentences:     {stats['total_sentences']}")
    print(f"Successfully matched: {stats['total_matched']}")
    print(f"Failed to match:     {stats['total_unmatched']}")
    print(f"Success rate:        {stats['success_rate']:.2f}%")
    
    # Word count statistics
    print("\n📝 WORD COUNT STATISTICS")
    print("-" * 80)
    print(f"Total words in sentences: {stats['sentence_word_counts']['total']}")
    print(f"Total words in segments:  {stats['segment_word_counts']['total']}")
    print(f"Word difference:          {abs(stats['sentence_word_counts']['total'] - stats['segment_word_counts']['total'])}")
    print(f"\nSentence word count - Min: {stats['sentence_word_counts']['min']}, "
          f"Max: {stats['sentence_word_counts']['max']}, "
          f"Avg: {stats['sentence_word_counts']['avg']:.1f}")
    print(f"Segment word count  - Min: {stats['segment_word_counts']['min']}, "
          f"Max: {stats['segment_word_counts']['max']}, "
          f"Avg: {stats['segment_word_counts']['avg']:.1f}")
    
    # Accuracy statistics
    print("\n🎯 ACCURACY STATISTICS")
    print("-" * 80)
    print(f"Average word difference:  {stats['word_differences']['avg']:.2f} words")
    print(f"Maximum word difference:  {stats['word_differences']['max']} words")
    print(f"Exact word matches:       {stats['word_differences']['exact_matches']} "
          f"({stats['word_differences']['exact_matches']/stats['total_matched']*100:.1f}%)")
    print(f"\nSimilarity - Min: {stats['similarities']['min']:.2%}, "
          f"Max: {stats['similarities']['max']:.2%}, "
          f"Avg: {stats['similarities']['avg']:.2%}")
    
    # Segment usage
    print("\n🔢 SEGMENT USAGE")
    print("-" * 80)
    print(f"Unique segments used:         {stats['segments_usage']['unique_segments_used']}")
    print(f"Total segment assignments:    {stats['segments_usage']['total_segments_used']}")
    print(f"Avg segments per sentence:    {stats['segments_usage']['avg_segments_per_sentence']:.2f}")
    print(f"Multi-segment matches:        {stats['multi_segment_matches']} "
          f"({stats['multi_segment_matches']/stats['total_matched']*100:.1f}%)")
    
    # Unmatched sentences
    if stats['unmatched_sentences']:
        print("\n❌ UNMATCHED SENTENCES")
        print("-" * 80)
        for item in stats['unmatched_sentences']:
            print(f"Line {item['line_number']}: {item['sentence'][:80]}...")
    
    # Sample matches
    print("\n📋 SAMPLE MATCHES (First 5)")
    print("-" * 80)
    for i, result in enumerate(results[:5]):
        if result.get('matched_segments'):
            print(f"\n{i+1}. Line {result['line_number']}: {result['sentence'][:60]}...")
            print(f"   → Segments {result['segment_ids']} "
                  f"({result['start_time']:.2f}s-{result['end_time']:.2f}s)")
            print(f"   Similarity: {result['similarity']:.2%}, "
                  f"Words: {result['word_count_sentence']}→{result['word_count_segments']}")
    
    print("\n" + "=" * 80)
    
    if stats['is_complete']:
        print("✅ VERIFICATION PASSED: All sentences successfully matched!")
    else:
        print("⚠️  VERIFICATION INCOMPLETE: Some sentences could not be matched")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Default file paths
    text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
    segments_file = "whisper_output/whisper_segments.json"
    output_file = "output/sentence_segment_matches.json"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
    if len(sys.argv) > 2:
        segments_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    print("🔍 Starting verification process...")
    print(f"Text file:     {text_file}")
    print(f"Segments file: {segments_file}")
    print(f"Output file:   {output_file}")
    print()
    
    # Perform matching
    results = match_sentences_to_segments(
        text_file_path=text_file,
        segments_file_path=segments_file,
        max_word_diff=2,
        min_similarity=0.5,
        max_search_window=50  # Limit search to prevent jumping to wrong matches
    )
    
    # Save results
    print("\n💾 Saving results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_file}")
    
    # Verify completeness
    print("\n🔍 Analyzing results...")
    stats = verify_matching_completeness(results)
    
    # Print report
    print()
    print_verification_report(stats, results)
    
    # Exit with appropriate code
    sys.exit(0 if stats['is_complete'] else 1)

