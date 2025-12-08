"""
METHOD W: Word-level sentence matching using whisper_words.json

This method matches sentences by finding word sequences in the word-level transcription,
which is more precise and reliable than segment-based matching.
"""

import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple


def normalize_word(word: str) -> str:
    """
    Normalize a word for matching (remove punctuation, lowercase).
    """
    # Remove punctuation and convert to lowercase
    word = re.sub(r'[^\w]', '', word.lower())
    return word


def extract_words_from_sentence(sentence: str) -> List[str]:
    """
    Extract and normalize words from a sentence.
    """
    # Split into words and normalize
    words = sentence.split()
    normalized = [normalize_word(w) for w in words if normalize_word(w)]
    return normalized


def calculate_word_sequence_similarity(text_words: List[str], audio_words: List[str]) -> float:
    """
    Calculate similarity between two word sequences.
    """
    if not text_words or not audio_words:
        return 0.0
    
    # Join words into strings for comparison
    text_str = " ".join(text_words)
    audio_str = " ".join(audio_words)
    
    return SequenceMatcher(None, text_str, audio_str).ratio()


def find_word_sequence(
    sentence_words: List[str],
    all_words: List[Dict],
    start_word_idx: int = 0,
    max_search_window: int = 500,
    min_similarity: float = 0.6,
    max_word_gap: int = 5,
    max_word_jump: Optional[int] = None,
    max_time_jump: Optional[float] = None,
    previous_end_time: Optional[float] = None
) -> Optional[Dict]:
    """
    Find a sequence of words in the word-level transcription.
    
    Args:
        sentence_words: Normalized words from the sentence
        all_words: List of all word dictionaries from whisper
        start_word_idx: Starting index for search
        max_search_window: Maximum number of words to search through
        min_similarity: Minimum similarity threshold
        max_word_gap: Maximum number of extra/missing words allowed
        max_word_jump: Maximum word index jump from start_word_idx (for strict sequential)
        max_time_jump: Maximum time jump from previous_end_time (for strict sequential)
        previous_end_time: End time of previous sentence (for sequential validation)
    
    Returns:
        Dictionary with match information or None
    """
    if not sentence_words:
        return None
    
    num_sentence_words = len(sentence_words)
    search_end = min(start_word_idx + max_search_window, len(all_words))
    
    best_match = None
    best_similarity = 0.0
    
    # Search for the word sequence
    for start_idx in range(start_word_idx, search_end):
        # STRICT SEQUENTIAL: Reject matches that jump too far ahead
        if max_word_jump is not None:
            word_jump = start_idx - start_word_idx
            if word_jump > max_word_jump:
                # All subsequent matches will also be too far ahead, break early
                break
        
        # Try different lengths around the expected number of words
        for length in range(
            max(1, num_sentence_words - max_word_gap),
            min(num_sentence_words + max_word_gap + 1, len(all_words) - start_idx + 1)
        ):
            end_idx = start_idx + length
            if end_idx > len(all_words):
                break
            
            # Extract audio words
            audio_words = [
                normalize_word(all_words[i]['word'])
                for i in range(start_idx, end_idx)
            ]
            
            # Calculate similarity
            similarity = calculate_word_sequence_similarity(sentence_words, audio_words)
            
            if similarity >= min_similarity and similarity > best_similarity:
                matched_words = all_words[start_idx:end_idx]
                match_start_time = matched_words[0]['start']
                
                # STRICT SEQUENTIAL: Reject matches with large time jumps
                if max_time_jump is not None and previous_end_time is not None:
                    time_jump = match_start_time - previous_end_time
                    if time_jump > max_time_jump:
                        # This match jumps too far ahead in time, skip it
                        continue
                
                best_similarity = similarity
                
                best_match = {
                    'matched_words': matched_words,
                    'word_indices': list(range(start_idx, end_idx)),
                    'start_word_idx': start_idx,
                    'end_word_idx': end_idx - 1,
                    'start_time': match_start_time,
                    'end_time': matched_words[-1]['end'],
                    'similarity': similarity,
                    'num_sentence_words': num_sentence_words,
                    'num_matched_words': length,
                    'word_diff': abs(num_sentence_words - length)
                }
        
        # Early exit if we found a very good match close to start
        if best_similarity > 0.95 and best_match:
            word_jump = best_match['start_word_idx'] - start_word_idx
            # If it's very close to start position and within sequential limits, we can stop early
            if word_jump < 20:
                break
    
    return best_match


def match_sentences_using_words(
    text_file_path: str,
    words_file_path: str,
    min_similarity: float = 0.6,
    max_word_gap: int = 5,
    max_search_window: int = 500,
    strict_sequential: bool = True,
    max_word_jump: int = 50,
    max_time_jump: float = 5.0
) -> List[Dict]:
    """
    Match sentences from text file using word-level timestamps.
    
    Args:
        text_file_path: Path to text file with sentences
        words_file_path: Path to whisper_words.json
        min_similarity: Minimum similarity threshold (0.6 = 60%)
        max_word_gap: Allow this many extra/missing words
        max_search_window: Search this many words ahead
        strict_sequential: Enforce strict sequential matching (prevent large jumps)
        max_word_jump: Maximum word index jump allowed (default: 50 words)
        max_time_jump: Maximum time jump allowed in seconds (default: 5.0s)
    
    Returns:
        List of match results for each sentence
    """
    # Load words
    print("Loading word-level transcription...")
    with open(words_file_path, 'r', encoding='utf-8') as f:
        all_words = json.load(f)
    print(f"  Loaded {len(all_words)} words")
    
    # Load text file
    print(f"Loading text file...")
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
    
    print(f"  Loaded {len(sentences)} sentences")
    print()
    
    # Match sentences
    results = []
    current_word_idx = 0
    previous_end_time = None
    
    for idx, sentence in enumerate(sentences):
        print(f"Matching sentence {idx + 1}/{len(sentences)}: {sentence[:60]}...")
        
        # Extract words from sentence
        sentence_words = extract_words_from_sentence(sentence)
        
        if not sentence_words:
            results.append({
                'line_number': idx + 1,
                'sentence': sentence,
                'matched_words': None,
                'error': 'No words in sentence'
            })
            print(f"  ⚠️  Empty sentence")
            continue
        
        # Find word sequence with strict sequential constraints
        match_result = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=max_search_window,
            min_similarity=min_similarity,
            max_word_gap=max_word_gap,
            max_word_jump=max_word_jump if strict_sequential else None,
            max_time_jump=max_time_jump if strict_sequential else None,
            previous_end_time=previous_end_time
        )
        
        if match_result:
            # Validate match is sequential (check for large jumps)
            word_jump = match_result['start_word_idx'] - current_word_idx
            time_jump = None
            if previous_end_time is not None:
                time_jump = match_result['start_time'] - previous_end_time
            
            # Warn if jump is large (even if within limits)
            if word_jump > 50:
                print(f"  ⚠️  Large word jump: {word_jump} words (from {current_word_idx} to {match_result['start_word_idx']})")
            if time_jump is not None and time_jump > 10.0:
                print(f"  ⚠️  Large time jump: {time_jump:.2f}s (from {previous_end_time:.2f}s to {match_result['start_time']:.2f}s)")
            
            results.append({
                'line_number': idx + 1,
                'sentence': sentence,
                'sentence_words': sentence_words,
                **match_result,
                'matched_text': ' '.join([w['word'] for w in match_result['matched_words']])
            })
            
            # Update search position
            current_word_idx = match_result['end_word_idx'] + 1
            previous_end_time = match_result['end_time']
            
            duration = match_result['end_time'] - match_result['start_time']
            print(f"  ✓ Words [{match_result['start_word_idx']}-{match_result['end_word_idx']}] "
                  f"Time: {match_result['start_time']:.2f}s-{match_result['end_time']:.2f}s "
                  f"({duration:.2f}s)")
            print(f"    Similarity: {match_result['similarity']:.2%}, "
                  f"Words: {match_result['num_sentence_words']}→{match_result['num_matched_words']}")
        else:
            # Initial match failed - try with relaxed parameters but still enforce sequential constraints
            print(f"  ✗ No match found, trying relaxed search (still sequential)...")
            
            # Try with wider search window and lower threshold
            # But STILL enforce sequential constraints to prevent cascade
            relaxed_match = find_word_sequence(
                sentence_words=sentence_words,
                all_words=all_words,
                start_word_idx=max(0, current_word_idx - 50),  # Look back a bit (sequential audio)
                max_search_window=min(300, len(all_words) - max(0, current_word_idx - 50)),  # Smaller window
                min_similarity=max(0.45, min_similarity - 0.15),  # Lower threshold but not too low
                max_word_gap=max(8, max_word_gap + 3),  # More tolerance but not excessive
                max_word_jump=max_word_jump if strict_sequential else None,  # STILL enforce word jump limit
                max_time_jump=max_time_jump if strict_sequential else None,  # STILL enforce time jump limit
                previous_end_time=previous_end_time
            )
            
            if relaxed_match:
                # Validate relaxed match is still sequential
                word_jump = relaxed_match['start_word_idx'] - current_word_idx
                time_jump = None
                if previous_end_time is not None:
                    time_jump = relaxed_match['start_time'] - previous_end_time
                
                # Warn if jump is large
                if word_jump > 50:
                    print(f"  ⚠️  Large word jump in relaxed search: {word_jump} words")
                if time_jump is not None and time_jump > 10.0:
                    print(f"  ⚠️  Large time jump in relaxed search: {time_jump:.2f}s")
                
                # Found with relaxed search
                results.append({
                    'line_number': idx + 1,
                    'sentence': sentence,
                    'sentence_words': sentence_words,
                    **relaxed_match,
                    'matched_text': ' '.join([w['word'] for w in relaxed_match['matched_words']]),
                    'relaxed_search': True  # Flag that this used relaxed parameters
                })
                
                # Check if we found it before current position (cascade detected)
                if relaxed_match['start_word_idx'] < current_word_idx:
                    print(f"  ⚠️  Found with relaxed search at words {relaxed_match['start_word_idx']}-{relaxed_match['end_word_idx']}")
                    print(f"    ⚠️  WARNING: Found BEFORE current position! (cascade detected)")
                    print(f"       Current: {current_word_idx}, Found at: {relaxed_match['start_word_idx']}")
                    print(f"       This means previous search passed the actual location!")
                
                current_word_idx = relaxed_match['end_word_idx'] + 1
                previous_end_time = relaxed_match['end_time']
                
                duration = relaxed_match['end_time'] - relaxed_match['start_time']
                print(f"  ✓ Found with relaxed search: Words [{relaxed_match['start_word_idx']}-{relaxed_match['end_word_idx']}]")
                print(f"    Time: {relaxed_match['start_time']:.2f}s-{relaxed_match['end_time']:.2f}s ({duration:.2f}s)")
                print(f"    Similarity: {relaxed_match['similarity']:.2%}")
            else:
                # Still no match - record as unmatched
                # Since audiobooks read sequentially, we should NOT skip much
                # Just advance by 1 word so next sentence can try from current position + 1
                results.append({
                    'line_number': idx + 1,
                    'sentence': sentence,
                    'sentence_words': sentence_words,
                    'matched_words': None,
                    'error': 'No match found'
                })
                print(f"  ✗ No match found even with relaxed search")
                
                # Very conservative skip: just 1 word forward
                # This ensures we don't miss sequential sentences
                # The next sentence will try from this position + 1
                if current_word_idx < len(all_words) - 1:
                    current_word_idx += 1
                    print(f"    → Advancing by 1 word to {current_word_idx} (conservative skip for sequential audio)")
                else:
                    print(f"    → At end of words, cannot advance")
    
    return results


def save_results(results: List[Dict], output_path: str):
    """Save matching results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    total = len(results)
    matched = sum(1 for r in results if r.get('matched_words') is not None)
    unmatched = total - matched
    
    if matched > 0:
        similarities = [r['similarity'] for r in results if r.get('similarity')]
        word_diffs = [r['word_diff'] for r in results if r.get('word_diff') is not None]
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_word_diff = sum(word_diffs) / len(word_diffs) if word_diffs else 0
        exact_matches = sum(1 for d in word_diffs if d == 0)
    else:
        avg_similarity = 0
        avg_word_diff = 0
        exact_matches = 0
    
    print("\n" + "=" * 80)
    print("METHOD W: WORD-LEVEL MATCHING SUMMARY")
    print("=" * 80)
    print(f"Total sentences:      {total}")
    print(f"Successfully matched: {matched} ({matched/total*100:.1f}%)")
    print(f"Failed to match:      {unmatched} ({unmatched/total*100:.1f}%)")
    
    if matched > 0:
        print(f"\nAverage similarity:   {avg_similarity:.2%}")
        print(f"Average word diff:    {avg_word_diff:.2f} words")
        print(f"Exact word matches:   {exact_matches} ({exact_matches/matched*100:.1f}%)")
    
    if unmatched > 0:
        print(f"\n⚠️  {unmatched} sentences could not be matched")
    else:
        print(f"\n✅ All sentences matched successfully!")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Default paths
    text_file = "data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.txt"
    words_file = "whisper_output/whisper_words.json"
    output_file = "output/word_level_matches.json"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
    if len(sys.argv) > 2:
        words_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    print("=" * 80)
    print("METHOD W: Word-Level Sentence Matching")
    print("=" * 80)
    print(f"Text file:  {text_file}")
    print(f"Words file: {words_file}")
    print(f"Output:     {output_file}")
    print("-" * 80)
    print()
    
    # Perform matching with strict sequential constraints
    results = match_sentences_using_words(
        text_file_path=text_file,
        words_file_path=words_file,
        min_similarity=0.6,
        max_word_gap=5,
        max_search_window=500,
        strict_sequential=True,  # Enforce strict sequential matching
        max_word_jump=50,  # Maximum 50 word jump
        max_time_jump=5.0  # Maximum 5 second jump
    )
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    # Exit code
    sys.exit(0 if all(r.get('matched_words') for r in results) else 1)

