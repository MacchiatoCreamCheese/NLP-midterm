# NLP-midterm

Vietnamese audiobook alignment: Whisper transcription, word-level matching, and per-sentence audio cutting.

## Installation

```bash
pip install -r requirements.txt
```

## Workflow (word-level)

1) Transcribe audio with Whisper and save word timestamps  
2) Match sentences to words  
3) Cut audio segments

One-shot pipeline:
```bash
python run_word_pipeline.py <audio_file> <text_file> \
  --whisper-root whisper_output_xaxoi \
  --output-root output_xaxoi \
  --model base \
  --padding 0.3
```

Outputs:
- `whisper_output_xaxoi/<book>/whisper_words.json` (and segments/transcription)
- `output_xaxoi/<book>_word_level_matches.json`
- `output_xaxoi/audio_segments_method_w/<book>/sentence_00001.wav` (+ .txt)

### Running steps manually (optional)

Transcribe only:
```bash
python src/align_vietnamese_audio.py transcribe <audio_file> --model base --output-dir whisper_output_xaxoi/<book>
```

Word matching:
```bash
python match_sentence_words.py <text_file> whisper_output_xaxoi/<book>/whisper_words.json output_xaxoi/<book>_word_level_matches.json
```

Cut from word matches:
```bash
python cut_audio_from_word_matches.py output_xaxoi/<book>_word_level_matches.json <audio_file> output_xaxoi/audio_segments_method_w/<book> 0.3
```

### Sentence splitting helper (optional)

```bash
python src/sentence_splitter.py <input_file> [-o output_file]
```

## Project Structure

```
NLP-midterm/
├── data/                          # Text and audio samples
├── src/
│   ├── sentence_splitter.py       # Split text into sentences
│   └── align_vietnamese_audio.py  # Whisper transcribe + cut helpers
├── match_sentence_words.py
├── cut_audio_from_word_matches.py
├── run_word_pipeline.py
├── README.md
└── requirements.txt
```

## Notes

- Recommended Whisper models: `base` or `small`
- Audio loading uses librosa (no ffmpeg required for common formats)

## DAISY 3 package generation

Generate `main.xml`, SMIL, OPF, and NCX from the word-level JSON files:

```bash
python scripts/generate_daisy.py \
  --json-dir output_xaxoi \
  --audio-dir data/Audio-XaXoiThonNguaGia \
  --out-dir build/daisy
```

Defaults are already set for the provided Xa Xôi Thôn Ngựa Già metadata. Audio is copied into `build/daisy/media` unless `--no-copy-media` is used. Add `--include-sentence-nav` to expose per-sentence navPoints in `navigation.ncx`.