"""
METHOD W: Word-level sentence matching using whisper_words.json

This method matches sentences by finding word sequences in the word-level transcription,
which is more precise and reliable than segment-based matching.
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple


def strip_accents(text: str) -> str:
    """Remove Vietnamese accents for number-word detection."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


NUMBER_WORDS = {
    # Without accents
    "khong",
    "mot",
    "hai",
    "ba",
    "bon",
    "nam",
    "sau",
    "bay",
    "tam",
    "chin",
    "muoi",
    "linh",
    "le",
    "tram",
    "ngan",
    "nghin",
    "trieu",
    "ty",
    # With common accents
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
    "mười",
    "mươi",
    "linh",
    "lẻ",
    "trăm",
    "ngàn",
    "nghìn",
    "triệu",
    "tỷ",
    "lăm",
}


def is_numeric_token(token: str) -> bool:
    """Check if token is numeric (integer or decimal, optional sign)."""
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d+)?", token))


def is_number_word(token: str) -> bool:
    """Check if token is a spelled-out number (rough Vietnamese coverage)."""
    if token in NUMBER_WORDS:
        return True
    plain = strip_accents(token)
    return plain in NUMBER_WORDS


def normalize_word(word: str) -> str:
    """
    Normalize a word for matching (remove punctuation, lowercase).
    Numbers (digit or spelled) are mapped to a placeholder to allow either form to match.
    """
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r"[^\w]", "", word.lower())
    if not cleaned:
        return ""
    if is_numeric_token(cleaned) or is_number_word(cleaned):
        return "numtok"
    return cleaned


def collapse_num_tokens(tokens: List[str]) -> List[str]:
    """Collapse consecutive numeric placeholders to align digit vs multi-word numbers."""
    collapsed: List[str] = []
    for tok in tokens:
        if tok == "numtok" and collapsed and collapsed[-1] == "numtok":
            continue
        collapsed.append(tok)
    return collapsed


def try_split_merged_word(word: str, min_part_len: int = 2) -> List[Tuple[str, str]]:
    """
    Try to split a merged Vietnamese word into two monosyllabic words.
    Returns list of (first_part, second_part) candidates.
    
    Vietnamese words are typically 1-4 characters. If a word is 5+ chars,
    it might be two merged words.
    """
    if len(word) < 4:  # Too short to be merged
        return []
    
    candidates = []
    # Try splits where each part is at least min_part_len and at most 4 chars
    # Vietnamese monosyllabic words are typically 1-4 characters
    for split_pos in range(min_part_len, min(len(word) - min_part_len + 1, 5)):
        first = word[:split_pos]
        second = word[split_pos:]
        # Both parts should be reasonable length (1-4 chars for Vietnamese)
        if 1 <= len(first) <= 4 and 1 <= len(second) <= 4:
            candidates.append((first, second))
    
    return candidates


def expand_audio_words_with_splits(audio_words: List[str], sentence_words: List[str]) -> List[List[str]]:
    """
    Generate candidate audio word sequences by trying to split merged words.
    Returns list of candidate sequences to try matching.
    
    Vietnamese is mostly monosyllabic (1-4 chars per word). If Whisper merged
    two words (e.g., "anh" + "kha" -> "ancha"), try splitting them.
    """
    if not audio_words or not sentence_words:
        return [audio_words]
    
    candidates = [audio_words]  # Always include original
    
    # Priority: split if audio sequence is shorter than sentence (likely merged words)
    if len(audio_words) < len(sentence_words):
        # Try splitting the longest words first
        word_lengths = [(len(w), i) for i, w in enumerate(audio_words)]
        word_lengths.sort(reverse=True)  # Longest first
        
        for word_len, i in word_lengths:
            if word_len >= 4:  # 4+ chars might be merged
                splits = try_split_merged_word(audio_words[i])
                for first, second in splits:
                    new_sequence = audio_words[:i] + [first, second] + audio_words[i+1:]
                    # Avoid duplicates
                    if new_sequence not in candidates:
                        candidates.append(new_sequence)
                        # Limit to avoid too many candidates
                        if len(candidates) >= 10:
                            break
                if len(candidates) >= 10:
                    break
    
    # Also try splitting any very long words (5+ chars) even if counts match
    for i, audio_word in enumerate(audio_words):
        if len(audio_word) >= 5:  # Very suspicious for Vietnamese
            splits = try_split_merged_word(audio_word)
            for first, second in splits:
                new_sequence = audio_words[:i] + [first, second] + audio_words[i+1:]
                if new_sequence not in candidates:
                    candidates.append(new_sequence)
                    if len(candidates) >= 15:  # Allow more for very long words
                        break
            if len(candidates) >= 15:
                break
    
    return candidates


def extract_words_from_sentence(sentence: str) -> List[str]:
    """
    Extract and normalize words from a sentence.
    """
    # Split into words and normalize
    words = sentence.split()
    normalized = []
    for w in words:
        nw = normalize_word(w)
        if nw:
            normalized.append(nw)
    normalized = collapse_num_tokens(normalized)
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
            audio_words_raw = [
                normalize_word(all_words[i]['word'])
                for i in range(start_idx, end_idx)
            ]
            audio_words_raw = [w for w in audio_words_raw if w]
            audio_words_raw = collapse_num_tokens(audio_words_raw)
            
            # Try original and split candidates
            audio_candidates = expand_audio_words_with_splits(audio_words_raw, sentence_words)
            
            # Try each candidate and pick the best similarity
            best_candidate_similarity = 0.0
            best_candidate_words = audio_words_raw
            
            for candidate in audio_candidates:
                candidate = collapse_num_tokens(candidate)
                sim = calculate_word_sequence_similarity(sentence_words, candidate)
                if sim > best_candidate_similarity:
                    best_candidate_similarity = sim
                    best_candidate_words = candidate
            
            similarity = best_candidate_similarity
            audio_words = best_candidate_words
            # Use the length of the best candidate (may be longer if words were split)
            actual_audio_length = len(best_candidate_words)
            
            if similarity >= min_similarity:
                matched_words = all_words[start_idx:end_idx]
                match_start_time = matched_words[0]['start']
                word_jump = start_idx - start_word_idx
                word_diff = abs(num_sentence_words - actual_audio_length)
                
                # Calculate composite score that prefers:
                # 1. High similarity (0-1)
                # 2. Exact word count match (word_diff = 0)
                # 3. Close proximity (small word_jump)
                # 4. Close time jump (if available)
                
                # Base score: similarity (0-1)
                score = similarity
                
                # Bonus for exact word count match (up to +0.3)
                word_count_bonus = max(0, 0.3 - (word_diff * 0.05))
                score += word_count_bonus
                
                # Penalty for word position jump (more than 10 words away)
                if word_jump > 10:
                    jump_penalty = min(0.5, (word_jump - 10) * 0.01)  # 0.01 per word over 10
                    score -= jump_penalty
                
                # Penalty for time jump (if we have previous time)
                if previous_end_time is not None:
                    time_jump = match_start_time - previous_end_time
                    # Allow reasonable gaps (pauses, music, etc) but penalize large jumps
                    if time_jump > 3.0:  # More than 3 seconds
                        time_penalty = min(0.3, (time_jump - 3.0) * 0.02)  # 0.02 per second over 3s
                        score -= time_penalty
                
                # HARD LIMIT: Reject matches with extreme jumps (likely wrong)
                if max_word_jump is not None and word_jump > max_word_jump:
                    continue
                if max_time_jump is not None and previous_end_time is not None:
                    time_jump = match_start_time - previous_end_time
                    if time_jump > max_time_jump:
                        continue
                
                # Update best match if this score is better
                if score > best_similarity:
                    best_similarity = score
                    
                    best_match = {
                        'matched_words': matched_words,
                        'word_indices': list(range(start_idx, end_idx)),
                        'start_word_idx': start_idx,
                        'end_word_idx': end_idx - 1,
                        'start_time': match_start_time,
                        'end_time': matched_words[-1]['end'],
                        'similarity': similarity,
                        'score': score,
                        'num_sentence_words': num_sentence_words,
                        'num_matched_words': length,
                        'word_diff': word_diff
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
    max_word_jump: int = 100,
    max_time_jump: float = 30.0
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
        max_word_jump: Maximum word index jump allowed (default: 100 words, hard limit)
        max_time_jump: Maximum time jump allowed in seconds (default: 30.0s, hard limit)
    
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
    previous_matched = True  # Track if previous sentence matched; only allow big jumps after a miss
    previous_similarity = None  # Track similarity of the last successful match
    consecutive_failures = 0  # Track consecutive failures for recovery
    
    for idx, sentence in enumerate(sentences):
        print(f"Matching sentence {idx + 1}/{len(sentences)}: {sentence[:60]}...")
        
        # Extract words from sentence
        sentence_words = extract_words_from_sentence(sentence)
        
        # Adjust parameters based on consecutive failures
        # After 2+ failures, allow slightly larger jumps but still conservative
        is_recovery_mode = consecutive_failures >= 2
        
        # MINIMIZE WORD JUMPS: Very conservative by default
        if previous_matched:
            # After a successful match, keep jumps very small (5-10 words max)
            if previous_similarity is not None and previous_similarity >= 0.8:
                # High confidence match: very tight (5 words)
                base_word_jump = 5
                base_time_jump = 8.0
            elif previous_similarity is not None and previous_similarity >= 0.6:
                # Medium confidence: still tight (8 words)
                base_word_jump = 8
                base_time_jump = 10.0
            else:
                # Lower confidence but still matched: moderate (10 words)
                base_word_jump = 10
                base_time_jump = 12.0
        else:
            # After unmatched sentence: allow small jump (15-20 words max)
            if is_recovery_mode:
                # Multiple failures: slightly more lenient but still small
                base_word_jump = 20
                base_time_jump = 15.0
            else:
                # Single failure: small jump
                base_word_jump = 15
                base_time_jump = 12.0
        
        # Per-sentence tuning
        is_short_sentence = len(sentence_words) <= 5
        is_long_sentence = len(sentence_words) >= 30
        sentence_min_similarity = max(0.45, min_similarity - 0.1) if is_short_sentence else min_similarity
        sentence_max_word_gap = max_word_gap + (1 if is_short_sentence else 0)
        if is_long_sentence:
            # Allow partial matches on long lines (missing tail) by relaxing thresholds and gaps
            sentence_min_similarity = max(0.4, min_similarity - 0.2)
            sentence_max_word_gap = max(sentence_max_word_gap, int(len(sentence_words) * 0.4))
        
        # Apply sentence-specific adjustments
        allowed_word_jump = base_word_jump
        allowed_time_jump = base_time_jump
        
        # Short sentences: even tighter
        if is_short_sentence and previous_matched:
            allowed_word_jump = min(allowed_word_jump, 6)
            allowed_time_jump = min(allowed_time_jump, 8.0)
        
        # Ensure we never exceed the hard limit
        allowed_word_jump = min(allowed_word_jump, max_word_jump)
        allowed_time_jump = min(allowed_time_jump, max_time_jump)
        
        # Search window: keep reasonable but not excessive
        if is_recovery_mode:
            sentence_max_search_window = min(max_search_window, 400)  # Reduced from 1500
        elif is_short_sentence:
            sentence_max_search_window = min(max_search_window, 150)
        else:
            sentence_max_search_window = max_search_window
        
        if not sentence_words:
            results.append({
                'line_number': idx + 1,
                'sentence': sentence,
                'matched_words': None,
                'error': 'No words in sentence'
            })
            print(f"  ⚠️  Empty sentence")
            consecutive_failures += 1
            continue
        
        # Find word sequence with strict sequential constraints
        match_result = find_word_sequence(
            sentence_words=sentence_words,
            all_words=all_words,
            start_word_idx=current_word_idx,
            max_search_window=sentence_max_search_window,
            min_similarity=sentence_min_similarity,
            max_word_gap=sentence_max_word_gap,
            max_word_jump=allowed_word_jump if strict_sequential else None,
            max_time_jump=allowed_time_jump if strict_sequential else None,
            previous_end_time=previous_end_time
        )
        
        if match_result:
            # Validate match quality - reject very poor matches
            if match_result['similarity'] < 0.35:
                print(f"  ✗ Match found but similarity too low ({match_result['similarity']:.2%}), rejecting...")
                match_result = None
            else:
                # Validate match is sequential (check for large jumps)
                word_jump = match_result['start_word_idx'] - current_word_idx
                time_jump = None
                if previous_end_time is not None:
                    time_jump = match_result['start_time'] - previous_end_time
                
                # Warn if jump is large (even if within limits)
                if word_jump > 10:
                    print(f"  ⚠️  Large word jump: {word_jump} words (from {current_word_idx} to {match_result['start_word_idx']})")
                if time_jump is not None and time_jump > 8.0:
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
                previous_matched = True
                previous_similarity = match_result['similarity']
                consecutive_failures = 0  # Reset failure counter on success
                
                duration = match_result['end_time'] - match_result['start_time']
                print(f"  ✓ Words [{match_result['start_word_idx']}-{match_result['end_word_idx']}] "
                      f"Time: {match_result['start_time']:.2f}s-{match_result['end_time']:.2f}s "
                      f"({duration:.2f}s)")
                score_info = f"Score: {match_result.get('score', match_result['similarity']):.3f}" if 'score' in match_result else ""
                print(f"    Similarity: {match_result['similarity']:.2%}, "
                      f"Words: {match_result['num_sentence_words']}→{match_result['num_matched_words']} "
                      f"({score_info})")
        else:
            # Initial match failed - try with relaxed parameters but still enforce sequential constraints
            print(f"  ✗ No match found, trying relaxed search (still sequential)...")
            
            # Try with wider search window and lower threshold
            # But STILL enforce sequential constraints to prevent cascade
            # Forward-only search to avoid grabbing the tail of the previous sentence
            relaxed_match = find_word_sequence(
                sentence_words=sentence_words,
                all_words=all_words,
                start_word_idx=current_word_idx,
                max_search_window=min(recovery_search_window if is_recovery_mode else (300 if not is_short_sentence else 150), len(all_words) - current_word_idx),  # Larger window in recovery
                min_similarity=max(0.35 if is_recovery_mode else 0.4, sentence_min_similarity - 0.1),  # Lower threshold in recovery
                max_word_gap=max(8, sentence_max_word_gap + 3),  # More tolerance but not excessive
                max_word_jump=allowed_word_jump if strict_sequential else None,  # STILL enforce word jump limit
                max_time_jump=allowed_time_jump if strict_sequential else None,  # STILL enforce time jump limit
                previous_end_time=previous_end_time
            )

            # Reject relaxed matches that start before our current cursor (prevents cascade)
            if relaxed_match and relaxed_match['start_word_idx'] < current_word_idx:
                relaxed_match = None
            
            # Reject very poor relaxed matches
            if relaxed_match and relaxed_match['similarity'] < 0.3:
                print(f"  ✗ Relaxed match found but similarity too low ({relaxed_match['similarity']:.2%}), rejecting...")
                relaxed_match = None
            
            if relaxed_match:
                # Validate relaxed match is still sequential
                word_jump = relaxed_match['start_word_idx'] - current_word_idx
                time_jump = None
                if previous_end_time is not None:
                    time_jump = relaxed_match['start_time'] - previous_end_time
                
                # Warn if jump is large
                if word_jump > 10:
                    print(f"  ⚠️  Large word jump in relaxed search: {word_jump} words")
                if time_jump is not None and time_jump > 8.0:
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
                previous_matched = True
                previous_similarity = relaxed_match['similarity']
                consecutive_failures = 0  # Reset failure counter on success
                
                duration = relaxed_match['end_time'] - relaxed_match['start_time']
                print(f"  ✓ Found with relaxed search: Words [{relaxed_match['start_word_idx']}-{relaxed_match['end_word_idx']}]")
                print(f"    Time: {relaxed_match['start_time']:.2f}s-{relaxed_match['end_time']:.2f}s ({duration:.2f}s)")
                score_info = f", Score: {relaxed_match.get('score', relaxed_match['similarity']):.3f}" if 'score' in relaxed_match else ""
                print(f"    Similarity: {relaxed_match['similarity']:.2%}{score_info}")
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
                
                consecutive_failures += 1
                previous_matched = False
                
                # In recovery mode (2+ failures), try a slightly more lenient search but still small jumps
                if is_recovery_mode:
                    # Try to find match with slightly relaxed parameters but STILL small jumps
                    print(f"    → Recovery mode: searching with relaxed similarity but small jumps...")
                    aggressive_match = find_word_sequence(
                        sentence_words=sentence_words,
                        all_words=all_words,
                        start_word_idx=current_word_idx,
                        max_search_window=min(400, len(all_words) - current_word_idx),  # Moderate window
                        min_similarity=0.35,  # Lower threshold
                        max_word_gap=10,  # More tolerance
                        max_word_jump=25,  # Still small jump (increased from 20 for recovery)
                        max_time_jump=20.0,  # Still reasonable time jump
                        previous_end_time=previous_end_time  # Still check time continuity
                    )
                    
                    if aggressive_match and aggressive_match['similarity'] >= 0.35:
                        word_jump = aggressive_match['start_word_idx'] - current_word_idx
                        if word_jump <= 25:  # Only accept if jump is still small
                            print(f"    ✓ Found match in recovery mode at word {aggressive_match['start_word_idx']} (jump: {word_jump})")
                            results[-1] = {
                                'line_number': idx + 1,
                                'sentence': sentence,
                                'sentence_words': sentence_words,
                                **aggressive_match,
                                'matched_text': ' '.join([w['word'] for w in aggressive_match['matched_words']]),
                                'recovery_mode': True
                            }
                            current_word_idx = aggressive_match['end_word_idx'] + 1
                            previous_end_time = aggressive_match['end_time']
                            previous_matched = True
                            previous_similarity = aggressive_match['similarity']
                            consecutive_failures = 0
                            continue
                        else:
                            print(f"    ✗ Recovery match found but jump too large ({word_jump} > 25), rejecting")
                
                # Very conservative skip: minimal advancement
                # This ensures we don't miss sequential sentences
                if current_word_idx < len(all_words) - 1:
                    # Even in recovery mode, skip very little (max 3 words)
                    skip_amount = min(3, len(sentence_words)) if is_recovery_mode else 1
                    current_word_idx = min(current_word_idx + skip_amount, len(all_words) - 1)
                    print(f"    → Advancing by {skip_amount} word(s) to {current_word_idx} ({'recovery mode' if is_recovery_mode else 'conservative skip'})")
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
    
    # Perform matching with scoring-based sequential matching
    results = match_sentences_using_words(
        text_file_path=text_file,
        words_file_path=words_file,
        min_similarity=0.6,
        max_word_gap=5,
        max_search_window=500,
        strict_sequential=True,  # Enforce sequential matching with scoring
        max_word_jump=100,  # Hard limit to reject truly wrong matches
        max_time_jump=30.0  # Hard limit to reject truly wrong matches
    )
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    # Exit code
    sys.exit(0 if all(r.get('matched_words') for r in results) else 1)

