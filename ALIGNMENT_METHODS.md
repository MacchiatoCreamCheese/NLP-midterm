# Audio Alignment Methods Comparison

## Problem with Current Basic Whisper Method

The basic Whisper alignment (`align_vietnamese_audio.py`) has low accuracy because:
1. **Word-level timestamps are approximate** - Whisper provides word timestamps but they're not forced-aligned
2. **Sentence matching is fuzzy** - The script tries to match transcribed words with text, but transcription may differ from exact text
3. **No forced alignment** - Basic Whisper doesn't force-align the known text to audio

## Better Solutions

### 1. WhisperX (Recommended) ⭐

**Why it's better:**
- Uses **forced alignment** with phoneme-level models
- More accurate word-level timestamps
- Better sentence boundary detection
- Still uses Whisper for transcription (good Vietnamese support)

**Installation:**
```bash
pip install whisperx
```

**Usage:**
```bash
python src/align_vietnamese_audio_whisperx.py "data/Audio-ThienThanNhoCuaToi/Track 1.mp3" "data/Text-ThienThanNhoCuaToi/Track 1.txt" output_whisperx base cpu
```

**Pros:**
- ✅ Better accuracy than basic Whisper
- ✅ Easy to use
- ✅ Works with Vietnamese
- ✅ No complex setup

**Cons:**
- ⚠️ Still not as accurate as MFA
- ⚠️ Requires alignment model download (automatic)

### 2. Montreal Forced Aligner (MFA) - Most Accurate 🏆

**Why it's best:**
- **Designed specifically for forced alignment**
- Uses acoustic models trained on Vietnamese
- Phoneme-level alignment
- Highest accuracy for matching text to audio

**Installation:**
```bash
conda install -c conda-forge montreal-forced-alignment
mfa model download acoustic vietnamese_mfa
mfa model download dictionary vietnamese_mfa
```

**Usage:**
```bash
# Prepare text file (one sentence per line, already done)
# Run MFA alignment
mfa align data/Audio-ThienThanNhoCuaToi/Track\ 1.mp3 data/Text-ThienThanNhoCuaToi/Track\ 1.txt vietnamese_mfa output_mfa/
```

**Pros:**
- ✅ Highest accuracy
- ✅ Professional-grade forced alignment
- ✅ Vietnamese acoustic models available

**Cons:**
- ⚠️ More complex setup
- ⚠️ Requires textgrid processing for cutting audio
- ⚠️ Command-line tool (less Python-friendly)

### 3. Aeneas (Alternative)

**Why consider it:**
- Lightweight forced alignment tool
- Works with many languages
- Good for basic alignment tasks

**Installation:**
```bash
pip install aeneas
```

**Note:** Aeneas may have limited Vietnamese support compared to MFA or WhisperX.

## Recommendation

1. **Start with WhisperX** - Best balance of accuracy and ease of use
2. **If accuracy still not good enough** - Try MFA (most accurate but more setup)
3. **For quick testing** - Use basic Whisper (fastest but least accurate)

## Expected Accuracy Improvements

- **Basic Whisper**: ~60-70% sentence alignment accuracy
- **WhisperX**: ~80-90% sentence alignment accuracy  
- **MFA**: ~90-95% sentence alignment accuracy

The accuracy depends on:
- Audio quality
- Speaker clarity
- Text matching (exact text vs. transcription differences)
- Audio/text synchronization


