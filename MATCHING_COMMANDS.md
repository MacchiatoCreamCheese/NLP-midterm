# Sentence-to-Segment Matching Commands

## ✅ Test Results

**Sentence 8 Test**: PASSED
- Matched to 7 segments [11-17]
- Word count: 58 → 58 (exact match)
- Similarity: 83.61%
- Time range: 37.92s - 54.98s

## 📋 Commands to Run

### 1. Run matching for the entire file (XaXoiThonNguaGiaP1.txt)
```bash
python verify_all_sentences.py
```

This will:
- Match all 493 sentences to Whisper segments
- Save results to `output/sentence_segment_matches.json`
- Show detailed verification report with statistics
- Check if all sentences were matched successfully

### 2. Run matching for a custom file
```bash
python verify_all_sentences.py <text_file> <segments_file> <output_file>
```

Example:
```bash
python verify_all_sentences.py data/Text-XaXoiThonNguaGia/XaXoiThonNguaGiaP2.txt whisper_output/whisper_segments.json output/matches_part2.json
```

### 3. Test a specific sentence (like sentence 8)
```bash
python test_sentence_8.py
```

### 4. Run quick examples
```bash
python example_match_usage.py
```

## 📊 What the Verification Script Checks

The `verify_all_sentences.py` script provides:

✅ **Completeness Check**: Are all sentences matched?
- Total sentences processed
- Success rate percentage
- List of any unmatched sentences

✅ **Word Count Statistics**: 
- Total words in sentences vs segments
- Min/Max/Average word counts
- Exact word matches count

✅ **Accuracy Metrics**:
- Average/Max word differences
- Min/Max/Average similarity scores

✅ **Segment Usage**:
- How many segments were used
- Average segments per sentence
- Multi-segment match statistics

## 🔧 Adjusting Parameters

You can modify matching behavior in the scripts:

```python
match_sentences_to_segments(
    text_file_path=text_file,
    segments_file_path=segments_file,
    max_word_diff=2,      # Allow ±2 word difference (default: 2)
    min_similarity=0.5    # Minimum 50% similarity (default: 0.6)
)
```

- **max_word_diff**: How many extra words allowed (1-3 recommended)
- **min_similarity**: Threshold for match quality (0.5-0.7 recommended)

## 📁 Output Files

After running, you'll get:
- `output/sentence_segment_matches.json` - Complete matching results
- Each entry contains:
  - Line number
  - Original sentence
  - Matched segments (with IDs, text, timestamps)
  - Word counts
  - Similarity score
  - Time range

