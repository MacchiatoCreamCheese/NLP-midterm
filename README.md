# NLP-midterm

Vietnamese text processing and audio alignment project.

## Features

- **Sentence Splitting**: Split Vietnamese text files into one sentence per line
- **Sentence Counting**: Count sentences in Vietnamese text files
- **Audio Alignment**: Align Vietnamese audio with text and cut into sentence segments

## Installation

```bash
pip install -r requirements.txt
```

**Note**: For audio processing, you also need `ffmpeg`:
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian)

## Usage

### Sentence Splitting

Split a text file into sentences (one per line):

```bash
python src/sentence_splitter.py <input_file> [-o output_file]
```

Example:
```bash
python src/sentence_splitter.py "data/Thiên thần nhỏ của tôi - Nguyễn Nhật Ánh.txt"
```

### Sentence Counting

Count sentences in the data files:

```bash
cd src
python -c "from main import count_sentences_in_data_files; count_sentences_in_data_files()"
```

### Audio Alignment

Align Vietnamese audio with text and cut into sentence segments:

```bash
python src/align_vietnamese_audio.py <audio_file> <sentences_file> [output_dir] [model]
```

Example:
```bash
python src/align_vietnamese_audio.py audio.wav "data/Thiên thần nhỏ của tôi - Nguyễn Nhật Ánh.processed.txt" output_audio base
```

**Parameters:**
- `audio_file`: Path to audio file (wav, mp3, etc.)
- `sentences_file`: Path to processed text file (one sentence per line)
- `output_dir`: Output directory for audio segments (default: `audio_segments`)
- `model`: Whisper model size - `tiny`, `base`, `small`, `medium`, `large` (default: `base`)

**Output:**
- `sentence_00001.wav`, `sentence_00002.wav`, ... - Audio segments
- `sentence_00001.txt`, `sentence_00002.txt`, ... - Corresponding text files
- `timestamps.txt` - Timestamps for each sentence
- `transcription.txt` - Full Whisper transcription

## Project Structure

```
NLP-midterm/
├── data/                          # Text files
│   ├── *.txt                      # Original text files
│   └── *.processed.txt            # Processed (one sentence per line)
├── src/
│   ├── sentence_splitter.py       # Sentence splitting script
│   ├── main.py                    # Main processing and counting
│   └── align_vietnamese_audio.py  # Audio alignment script
└── requirements.txt               # Python dependencies
```

## Notes

- All text files use UTF-8 encoding to support Vietnamese diacritics
- The audio alignment uses Whisper with Vietnamese language support
- For best results, use `base` or `small` Whisper models for Vietnamese