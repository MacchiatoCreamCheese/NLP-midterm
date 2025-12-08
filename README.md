# NLP-midterm

Vietnamese audio alignment with Whisper: transcribe, align to text, and cut sentence-level audio.

## Features

- Sentence splitting helper to prepare one-sentence-per-line text files
- Whisper transcription with saved segments/words
- Alignment of sentences to word-level timestamps
- Audio cutting to per-sentence WAVs and text

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Sentence splitting (optional)

```bash
python src/sentence_splitter.py <input_file> [-o output_file]
```

### Core workflows (`src/align_vietnamese_audio.py`)

Transcribe only (saves Whisper outputs for reuse):
```bash
python src/align_vietnamese_audio.py transcribe <audio_file> --model base --output-dir whisper_output
```

Align from saved Whisper results:
```bash
python src/align_vietnamese_audio.py align whisper_output --sentences-file sentences.txt --output-dir audio_segments
# or use detected Whisper segments instead of a text file
python src/align_vietnamese_audio.py align whisper_output --use-detected --output-dir audio_segments
```

Full run: transcribe, align, and cut in one step:
```bash
python src/align_vietnamese_audio.py full <audio_file> sentences.txt --model base --output-dir audio_segments
# use detected sentences (no text file)
python src/align_vietnamese_audio.py full <audio_file> --use-detected --output-dir audio_segments
```

Outputs:
- `sentence_00001.wav` / `sentence_00001.txt` etc.
- `timestamps.txt` and `transcription.txt`
- Saved Whisper results in `<audio>_whisper_results/`

## Project Structure

```
NLP-midterm/
├── data/                    # Text and audio samples
├── src/
│   ├── sentence_splitter.py # Split text into sentences
│   └── align_vietnamese_audio.py # Whisper transcription + alignment + cutting
├── README.md
└── requirements.txt         # Python dependencies
```

## Notes

- UTF-8 text files recommended for Vietnamese diacritics
- Whisper models `base` or `small` are good starting points
- Audio loading uses librosa and does not require ffmpeg for common formats