# Alignment Scoring System

## Problem

The original alignment code had two issues:
1. **Too loose limits** (50 words, 5 seconds) allowed large jumps that skipped content
2. **Too strict limits** (20 words, 2 seconds) caused alignment to fail early

Example: Sentence 46 in "Cố Vinh, người xứ lạ" jumped 73 words and 27 seconds ahead.

## Solution: Scoring-Based Matching

Instead of hard rejection limits, we now use a **composite scoring system** that prefers good matches while allowing flexibility when needed.

### Scoring Formula

```
score = similarity + word_count_bonus - jump_penalty - time_penalty
```

**Components:**

1. **Base Similarity** (0-1): Text-to-audio word sequence similarity
   
2. **Word Count Bonus** (+0 to +0.3):
   - Exact match (word_diff = 0): +0.30
   - 1 word off: +0.25
   - 2 words off: +0.20
   - etc.

3. **Jump Penalty** (-0 to -0.5):
   - First 10 words: no penalty
   - After 10 words: -0.01 per word
   - Example: 30-word jump = -0.20 penalty

4. **Time Penalty** (-0 to -0.3):
   - First 3 seconds: no penalty
   - After 3 seconds: -0.02 per second
   - Example: 13-second gap = -0.20 penalty

### Hard Limits (Safety Net)

- **max_word_jump**: 100 words (rejects truly wrong matches)
- **max_time_jump**: 30 seconds (rejects truly wrong matches)

These hard limits catch extreme cases but are rarely triggered.

## Benefits

1. ✅ **Prefers close matches**: Small jumps get higher scores
2. ✅ **Prefers exact word counts**: Matching word count is rewarded
3. ✅ **Allows reasonable gaps**: Pauses, music, narrator comments don't break alignment
4. ✅ **Flexible but controlled**: Adapts to content while preventing cascading errors

## Example Scoring

**Scenario**: Looking for 4-word sentence at word position 740

| Match | Similarity | Words | Jump | Time | Score | Winner? |
|-------|-----------|-------|------|------|-------|---------|
| A     | 0.94      | 4→4   | 2    | 0.5s | 1.24  | ✅ YES  |
| B     | 0.90      | 4→6   | 73   | 27s  | 0.67  | ❌ NO   |
| C     | 0.85      | 4→3   | 8    | 1.2s | 1.10  | ❌ NO   |

Match A wins despite lower raw similarity because:
- Exact word count: +0.30
- Close proximity: -0.00 (within 10 words)
- Short time gap: -0.00 (within 3 seconds)

## Configuration

Default parameters in all scripts:
```python
match_sentences_using_words(
    min_similarity=0.6,
    max_word_gap=5,
    max_search_window=500,
    strict_sequential=True,
    max_word_jump=100,    # Hard limit
    max_time_jump=30.0    # Hard limit
)
```

## Updated Files

- `match_sentence_words.py` - Core scoring logic
- `process_all_xaxoi_files.py` - Main processing pipeline
- `test_method_w.py` - Test script



