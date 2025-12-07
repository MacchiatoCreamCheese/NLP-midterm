# Method W: Audio Cutting from Word-Level Matches

This script cuts audio segments based on `word_level_matches.json` (Method W results).

## Features

✅ **Uses precise word-level timestamps** for matched sentences  
✅ **Fills gaps automatically** for unmatched sentences  
✅ **Handles edge cases** (start/end of audio, consecutive unmatched sentences)  

## How It Works

1. **Matched Sentences**: Uses `start_time` and `end_time` from the word-level match
2. **Unmatched Sentences**: Fills the gap between:
   - Previous matched sentence's `end_time` 
   - Next matched sentence's `start_time`
3. **Edge Cases**:
   - If no previous match: Uses 2 seconds ending at next match
   - If no next match: Uses 2 seconds starting after previous match
   - Ensures minimum duration of 0.5 seconds for all segments

## Usage

```bash
python cut_audio_from_word_matches.py <word_matches_json> <audio_file> <output_dir> [padding]
```

### Parameters

- `word_matches_json`: Path to `word_level_matches.json` file
- `audio_file`: Path to input audio file (MP3, WAV, etc.)
- `output_dir`: Directory to save cut audio segments
- `padding`: Optional padding in seconds (default: 0.3s) - added before and after each segment

### Example

```bash
python cut_audio_from_word_matches.py output/word_level_matches.json data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3 output/audio_segments_method_w 0.3
```

## Output

The script creates:
- `sentence_00001.wav`, `sentence_00002.wav`, ... - Audio segments
- `sentence_00001.txt`, `sentence_00002.txt`, ... - Text files for each segment

## Statistics

The script reports:
- Number of segments from matched sentences
- Number of segments from filled gaps
- Number of segments using default duration
- Number of skipped segments (no timing available)

## Example Output

```
================================================================================
Cutting Audio from Word-Level Matches (Method W)
================================================================================
Word matches file: output/word_level_matches.json
Audio file:        data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3
Output directory:  output/audio_segments_method_w
Padding:           0.3 seconds

Loading word matches from output/word_level_matches.json...
Loaded 493 sentence matches
  Matched:   483
  Unmatched: 10

Processing matches for audio cutting...
  Line 57: Filled gap from 381.52s to 382.06s (0.54s)
  ...

Created 493 audio segments to cut

Cutting audio from data/Audio-XaXoiThonNguaGia/XaXoiThonNguaGiaP1.mp3...
✓ Complete! Audio segments saved to: output/audio_segments_method_w
```

