import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation and converting to lowercase.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_words(text: str) -> int:
    """
    Count words in a text string.
    """
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len(normalized.split())


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts.
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def match_sentence_to_segments(
    sentence: str,
    segments: List[Dict],
    start_segment_id: int = 0,
    max_word_diff: int = 2,
    min_similarity: float = 0.6,
    max_search_window: int = 50
) -> Optional[Dict]:
    """
    Match a sentence from the text file to multiple consecutive Whisper segments.
    
    Args:
        sentence: The sentence to match
        segments: List of all Whisper segments
        start_segment_id: ID of segment to start searching from (for sequential matching)
        max_word_diff: Maximum allowed word count difference (default: 2)
        min_similarity: Minimum similarity threshold for matching (default: 0.6)
        max_search_window: Maximum number of segments to search forward (default: 50)
    
    Returns:
        Dictionary containing:
            - 'matched_segments': List of matched segment dictionaries
            - 'sentence': Original sentence
            - 'combined_text': Combined text from all matched segments
            - 'word_count_sentence': Word count in sentence
            - 'word_count_segments': Word count in combined segments
            - 'similarity': Similarity score
            - 'start_time': Start time of first segment
            - 'end_time': End time of last segment
        Returns None if no good match is found
    """
    sentence_word_count = count_words(sentence)
    
    if sentence_word_count == 0:
        return None
    
    # Normalize sentence for comparison
    normalized_sentence = normalize_text(sentence)
    sentence_words = normalized_sentence.split()
    
    best_match = None
    best_similarity = 0.0
    
    # Try different starting positions and lengths (with limited search window)
    search_end = min(start_segment_id + max_search_window, len(segments))
    
    for start_idx in range(start_segment_id, search_end):
        accumulated_text = ""
        accumulated_segments = []
        accumulated_word_count = 0
        
        # Try accumulating segments
        for end_idx in range(start_idx, min(start_idx + 20, len(segments))):  # Max 20 segments per match
            segment = segments[end_idx]
            accumulated_segments.append(segment)
            accumulated_text += " " + segment["text"]
            accumulated_word_count = count_words(accumulated_text)
            
            # Check if word count is within acceptable range
            if accumulated_word_count >= sentence_word_count - 1 and \
               accumulated_word_count <= sentence_word_count + max_word_diff:
                
                # Calculate similarity
                similarity = calculate_similarity(sentence, accumulated_text)
                
                if similarity >= min_similarity and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'matched_segments': accumulated_segments.copy(),
                        'sentence': sentence,
                        'combined_text': accumulated_text.strip(),
                        'word_count_sentence': sentence_word_count,
                        'word_count_segments': accumulated_word_count,
                        'similarity': similarity,
                        'start_time': accumulated_segments[0]['start'],
                        'end_time': accumulated_segments[-1]['end'],
                        'segment_ids': [s['id'] for s in accumulated_segments]
                    }
            
            # Stop if we've accumulated too many words
            if accumulated_word_count > sentence_word_count + max_word_diff + 5:
                break
    
    return best_match


def match_sentences_to_segments(
    text_file_path: str,
    segments_file_path: str,
    max_word_diff: int = 2,
    min_similarity: float = 0.6,
    max_search_window: int = 50
) -> List[Dict]:
    """
    Match all sentences from a text file to Whisper segments.
    
    Args:
        text_file_path: Path to the text file with sentences
        segments_file_path: Path to the whisper_segments.json file
        max_word_diff: Maximum allowed word count difference
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of match results for each sentence
    """
    # Load segments
    with open(segments_file_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # Load text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove line numbers and empty lines
    sentences = []
    for line in lines:
        # Remove line number prefix (format: "   123|text")
        if '|' in line:
            sentence = line.split('|', 1)[1].strip()
        else:
            sentence = line.strip()
        
        if sentence:  # Only include non-empty sentences
            sentences.append(sentence)
    
    # Match sentences sequentially
    results = []
    current_segment_id = 0
    
    for idx, sentence in enumerate(sentences):
        print(f"Matching sentence {idx + 1}/{len(sentences)}: {sentence[:50]}...")
        
        match_result = match_sentence_to_segments(
            sentence=sentence,
            segments=segments,
            start_segment_id=current_segment_id,
            max_word_diff=max_word_diff,
            min_similarity=min_similarity,
            max_search_window=max_search_window
        )
        
        if match_result:
            results.append({
                'line_number': idx + 1,
                **match_result
            })
            # Update starting position for next search
            current_segment_id = match_result['segment_ids'][-1] + 1
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


def save_matching_results(results: List[Dict], output_path: str):
    """
    Save matching results to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Match sentences from the text file to Whisper segments
    text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
    segments_file = "whisper_output/whisper_segments.json"
    output_file = "output/sentence_segment_matches.json"
    
    print("Starting sentence-to-segment matching...")
    print(f"Text file: {text_file}")
    print(f"Segments file: {segments_file}")
    print("-" * 80)
    
    results = match_sentences_to_segments(
        text_file_path=text_file,
        segments_file_path=segments_file,
        max_word_diff=2,
        min_similarity=0.6
    )
    
    # Save results
    save_matching_results(results, output_file)
    
    # Print summary
    successful_matches = sum(1 for r in results if r.get('matched_segments') is not None)
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total sentences: {len(results)}")
    print(f"  Successfully matched: {successful_matches}")
    print(f"  Failed to match: {len(results) - successful_matches}")
    print(f"  Success rate: {successful_matches/len(results)*100:.1f}%")

