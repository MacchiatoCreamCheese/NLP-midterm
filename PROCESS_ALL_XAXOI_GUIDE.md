# Processing All XaXoiThonNguaGia Files - Complete Guide

This guide explains how to process all audio/text files in the XaXoiThonNguaGia dataset using Method W.

## Overview

The processing pipeline consists of 3 steps:
1. **Whisper Transcription** - Transcribe audio to get word-level timestamps
2. **Method W Matching** - Match text sentences to word-level timestamps
3. **Audio Cutting** - Cut audio into sentence segments based on matches

## Files Created

### Scripts
- `process_all_xaxoi_files.py` - Interactive version (asks for confirmation)
- `process_all_xaxoi_batch.py` - Batch version (no interaction, good for automation)

### Output Directories
- `whisper_output_xaxoi/` - Whisper transcription results (one folder per audio file)
- `output_xaxoi/` - Method W matching results (one JSON file per file pair)
- `output_xaxoi/audio_segments_method_w/` - Cut audio segments (one folder per file pair)

## Usage

### Option 1: Interactive Processing

```bash
python process_all_xaxoi_files.py
```

This will:
- Show all found file pairs
- Ask for confirmation before starting
- Process each file step-by-step
- Show progress and results

### Option 2: Batch Processing (Recommended)

```bash
python process_all_xaxoi_batch.py
```

This will:
- Process all files automatically
- Show progress for each file
- Provide summary at the end

### Option 3: Custom Options

```bash
# Use a different Whisper model
python process_all_xaxoi_batch.py --model small

# Change audio padding
python process_all_xaxoi_batch.py --padding 0.5

# Skip steps (if already completed)
python process_all_xaxoi_batch.py --skip-whisper  # If Whisper already done
python process_all_xaxoi_batch.py --skip-matching  # If matching already done
python process_all_xaxoi_batch.py --skip-cutting  # If cutting already done

# Combine options
python process_all_xaxoi_batch.py --model small --padding 0.3 --skip-whisper
```

## Files Processed

The script processes these 8 file pairs:

1. **XaXoiThonNguaGiaP1** - Part 1
2. **XaXoiThonNguaGiaP2** - Part 2
3. **XaXoiThonNguaGiaP3** - Part 3
4. **Cánh bướm tím** - Short story
5. **Cố Vinh, người xứ lạ** - Short story
6. **NguoiKhoNhatTranGian** - Short story
7. **Seo Ly, kẻ khuấy động tình trường** - Short story
8. **Thắp một tuần hương** - Short story

## Output Structure

After processing, you'll have:

```
whisper_output_xaxoi/
├── XaXoiThonNguaGiaP1/
│   ├── whisper_segments.json
│   ├── whisper_words.json          ← Used by Method W
│   └── whisper_transcription.txt
├── XaXoiThonNguaGiaP2/
│   └── ...
└── ...

output_xaxoi/
├── XaXoiThonNguaGiaP1_word_level_matches.json  ← Method W results
├── XaXoiThonNguaGiaP2_word_level_matches.json
└── ...

output_xaxoi/audio_segments_method_w/
├── XaXoiThonNguaGiaP1/
│   ├── sentence_00001.wav
│   ├── sentence_00001.txt
│   ├── sentence_00002.wav
│   ├── sentence_00002.txt
│   └── ...
├── XaXoiThonNguaGiaP2/
│   └── ...
└── ...
```

## Processing Time

**Estimated times** (depending on your hardware):
- **Whisper**: 5-20 minutes per audio file (model-dependent)
- **Method W Matching**: 1-5 minutes per file
- **Audio Cutting**: 1-3 minutes per file

**Total for all 8 files**: Approximately 1-3 hours depending on:
- Whisper model size (base, small, medium, large)
- Audio file lengths
- CPU/GPU performance

## Troubleshooting

### Issue: "No matching text file found"
- Check that text files exist in `data/Text-XaXoiThonNguaGia/`
- Verify filenames match (case-sensitive on some systems)

### Issue: Whisper fails
- Check that you have whisper installed: `pip install openai-whisper`
- Try a smaller model: `--model base` or `--model tiny`
- Check audio file format is supported (MP3, WAV, etc.)

### Issue: Out of memory during Whisper
- Use a smaller model: `--model base` or `--model tiny`
- Process files one at a time manually

### Issue: Method W matching fails
- Check that `whisper_words.json` was created successfully
- Verify the text file format is correct
- Check for encoding issues (files should be UTF-8)

### Issue: Audio cutting fails
- Check that `word_level_matches.json` exists
- Verify audio file is accessible
- Check available disk space

## Resuming Interrupted Processing

If processing is interrupted, you can resume:

```bash
# If Whisper was completed but matching failed:
python process_all_xaxoi_batch.py --skip-whisper

# If matching was completed but cutting failed:
python process_all_xaxoi_batch.py --skip-whisper --skip-matching
```

The scripts automatically skip steps that are already completed.

## Quick Start

```bash
# Process all files (recommended)
python process_all_xaxoi_batch.py

# Or if you prefer interactive mode:
python process_all_xaxoi_files.py
```

That's it! The script will handle everything automatically. ☕

