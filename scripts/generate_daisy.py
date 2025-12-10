#!/usr/bin/env python3
"""Generate DAISY 3 package files (DTBook XML, SMIL, OPF, NCX) from JSON.

The script expects the word-level JSON files produced by the pipeline to live
in `output_xaxoi/` and the corresponding MP3s to live in
`data/Audio-XaXoiThonNguaGia/`. It writes DAISY outputs to `build/daisy/` by
default:
  - main.xml            (DTBook)
  - main.opf            (package/manifest)
  - navigation.ncx      (TOC)
  - smil/<chapter>.smil (one per chapter)
  - media/*             (audio + cover, when copying is enabled)

If sentence-level WAV segments already exist at
`output_xaxoi/audio_segments_method_w/<chapter>/sentence_00001.wav`, the
generator can use them when `--use-sentence-audio` is passed, so SMIL plays the
pre-cut clips instead of slicing the chapter MP3s by timecodes.

Usage (default metadata prefilled from user-provided values):
  python scripts/generate_daisy.py

You can override folders or metadata, and you can disable media copying:
  python scripts/generate_daisy.py --out-dir build/daisy --no-copy-media
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


DTBOOK_NS = "http://www.daisy.org/z3986/2005/dtbook/"
SMIL_NS = "http://www.w3.org/2001/SMIL20/"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"
OPF_NS = "http://openebook.org/namespaces/oeb-package/1.0/"
DC_NS = "http://purl.org/dc/elements/1.1/"


@dataclass
class Metadata:
    title: str
    creator: str
    subject: str
    description: str
    publisher: str
    date: str
    language: str
    identifier: str
    source_isbn: str
    source_url: Optional[str] = None
    note: Optional[str] = None
    reader: Optional[str] = None
    collector: Optional[str] = None


@dataclass
class Sentence:
    line_number: int
    text: str
    start: Optional[float]
    end: Optional[float]
    sid: str  # unique sentence ID shared with SMIL
    audio_path: Optional[Path] = None


@dataclass
class Chapter:
    json_path: Path
    audio_path: Path
    title: str
    cid: str  # chapter ID
    sentences: List[Sentence]
    smil_name: str  # filename (not full path) of SMIL


def normalize_name(value: str) -> str:
    """Lowercase string with non-alphanumerics stripped (for loose matching)."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def discover_audio_map(audio_dir: Path) -> Dict[str, Path]:
    """Return a mapping of normalized stem -> audio file path."""
    audio_map: Dict[str, Path] = {}
    for mp3 in sorted(audio_dir.glob("*.mp3")):
        stem_norm = normalize_name(mp3.stem)
        audio_map[stem_norm] = mp3
    return audio_map


def stem_to_audio(json_stem: str, audio_map: Dict[str, Path]) -> Path:
    """Resolve JSON stem to an MP3 Path using normalization and P1 fallback."""
    norm = normalize_name(json_stem)
    if norm in audio_map:
        return audio_map[norm]

    # Handle part-less stems that map to P1 audio (e.g., ...Gia -> ...GiaP1).
    p1_norm = f"{norm}p1"
    if p1_norm in audio_map:
        return audio_map[p1_norm]

    raise KeyError(
        f"No audio match for stem '{json_stem}'. Checked normalized keys: {list(audio_map.keys())}"
    )


def humanize_chapter_title(stem: str) -> str:
    """Derive a readable chapter title from a file stem."""
    # Replace underscores with spaces and collapse multiple separators.
    cleaned = re.sub(r"[_\s]+", " ", stem).strip()
    # Normalize casing but keep existing capital letters (title-case).
    return cleaned.title()


def format_npt(value: float) -> str:
    """Format a float as SMIL npt time with millisecond precision."""
    return f"npt={value:.3f}s"


def format_smil_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm for SMIL clipBegin/clipEnd attributes (H not zero-padded)."""
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def format_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS (rounded to nearest second)."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def audio_media_type(path: Path) -> str:
    """Return the appropriate media-type string for a given audio file."""
    ext = path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".m4a":
        return "audio/mp4"
    return "audio/mpeg"


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio file duration in seconds. Returns None if librosa is not available."""
    if not HAS_LIBROSA:
        return None
    try:
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}", file=sys.stderr)
        return None


def load_sentences(json_path: Path, chap_idx: int, audio_path: Optional[Path] = None) -> List[Sentence]:
    """Load sentences with timing, filling gaps like cut_audio_from_word_matches.py."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, object]] = []
    for entry in data:
        line_no = int(entry["line_number"])
        text = (entry.get("sentence") or "").strip()
        start_val = entry.get("start_time", entry.get("start"))
        end_val = entry.get("end_time", entry.get("end"))
        matched_ok = (
            entry.get("matched_words") is not None
            and "error" not in entry
            and start_val is not None
            and end_val is not None
        )
        records.append(
            {
                "line": line_no,
                "text": text,
                "start": float(start_val) if start_val is not None else None,
                "end": float(end_val) if end_val is not None else None,
                "matched": bool(matched_ok),
            }
        )

    sentences: List[Sentence] = []
    for i, rec in enumerate(records):
        if rec["matched"] and rec["start"] is not None and rec["end"] is not None:
            start = rec["start"]
            end = rec["end"]
        else:
            # Fill using neighbors following audio cutter rules.
            prev_end = None
            next_start = None

            for j in range(i - 1, -1, -1):
                prev = records[j]
                if prev["matched"] and prev["end"] is not None:
                    prev_end = prev["end"]
                    break
            for j in range(i + 1, len(records)):
                nxt = records[j]
                if nxt["matched"] and nxt["start"] is not None:
                    next_start = nxt["start"]
                    break

            if prev_end is not None and next_start is not None:
                start = prev_end
                end = next_start
                # Enforce minimum duration 0.5s; expand symmetrically if needed.
                if end - start < 0.5:
                    center = (start + end) / 2
                    start = max(0.0, center - 0.25)
                    end = center + 0.25
            elif prev_end is not None:
                start = prev_end
                end = start + 2.0
            elif next_start is not None:
                end = next_start
                start = max(0.0, end - 2.0)
            else:
                print(f"Warning: skipping line {rec['line']} in {json_path} (no adjacent timings)", file=sys.stderr)
                continue

        if end <= start:
            end = start + 0.01

        sid = f"s{chap_idx:02d}_{rec['line']:05d}"
        sentences.append(Sentence(rec["line"], rec["text"], start, end, sid))
    
    # For the last sentence, if it's the last one and we have audio, use actual audio duration
    if sentences and audio_path and HAS_LIBROSA:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration is not None and sentences[-1].end is not None:
            # If last sentence end is close to or beyond audio duration, cap it at audio duration
            if sentences[-1].end >= audio_duration * 0.95:  # Within 5% of end
                sentences[-1].end = audio_duration

    # Warn on non-monotonic sequences but keep going.
    last_end = 0.0
    for s in sentences:
        if s.start is not None and s.start < last_end:
            print(f"Warning: non-monotonic timing near line {s.line_number} in {json_path}", file=sys.stderr)
        if s.end is not None:
            last_end = s.end
    return sentences


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def indent(tree: ET.ElementTree) -> None:
    """Apply in-place indentation for pretty XML output."""
    try:
        ET.indent(tree, space="  ")  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for very old Python versions.
        def _indent(elem, level=0):
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                for child in elem:
                    _indent(child, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i

        _indent(tree.getroot())


def build_dtbook(meta: Metadata, chapters: List[Chapter], out_path: Path) -> None:
    ET.register_namespace("", DTBOOK_NS)
    root = ET.Element(f"{{{DTBOOK_NS}}}dtbook", {"version": "2005-3", "xml:lang": meta.language})

    head = ET.SubElement(root, f"{{{DTBOOK_NS}}}head")
    for name, value in (
        ("dc:Title", meta.title),
        ("dc:Creator", meta.creator),
        ("dc:Subject", meta.subject),
        ("dc:Publisher", meta.publisher),
        ("dc:Date", meta.date),
        ("dc:Language", meta.language),
        ("dc:Identifier", meta.identifier),
    ):
        ET.SubElement(head, f"{{{DTBOOK_NS}}}meta", {"name": name, "content": value})

    book = ET.SubElement(root, f"{{{DTBOOK_NS}}}book", {"showin": "blp"})

    # Per DTBook 2005-3, doctitle/docauthor live in frontmatter, not head.
    frontmatter = ET.SubElement(book, f"{{{DTBOOK_NS}}}frontmatter")
    doctitle = ET.SubElement(frontmatter, f"{{{DTBOOK_NS}}}doctitle")
    doctitle.text = meta.title
    docauthor = ET.SubElement(frontmatter, f"{{{DTBOOK_NS}}}docauthor")
    docauthor.text = meta.creator

    bodymatter = ET.SubElement(book, f"{{{DTBOOK_NS}}}bodymatter")

    for chapter in chapters:
        level = ET.SubElement(bodymatter, f"{{{DTBOOK_NS}}}level1", {"id": chapter.cid})
        h1 = ET.SubElement(level, f"{{{DTBOOK_NS}}}h1", {"smilref": f"smil/{chapter.smil_name}#seq_{chapter.cid}"})
        h1.text = chapter.title
        for sentence in chapter.sentences:
            p_attrs = {"id": sentence.sid, "smilref": f"smil/{chapter.smil_name}#par_{sentence.sid}"}
            ET.SubElement(level, f"{{{DTBOOK_NS}}}p", p_attrs).text = sentence.text

    tree = ET.ElementTree(root)
    indent(tree)
    doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE dtbook PUBLIC "-//NISO//DTD dtbook 2005-3//EN"\n'
        '  "http://www.daisy.org/z3986/2005/dtbook-2005-3.dtd">\n'
    )
    with out_path.open("wb") as f:
        f.write(doctype.encode("utf-8"))
        tree.write(f, encoding="utf-8", xml_declaration=False)


def build_smil(chapter: Chapter, main_href: str, out_path: Path, meta: Metadata) -> None:
    ET.register_namespace("", SMIL_NS)
    root = ET.Element(f"{{{SMIL_NS}}}smil")

    head = ET.SubElement(root, f"{{{SMIL_NS}}}head")
    ET.SubElement(head, f"{{{SMIL_NS}}}meta", {"name": "dtb:uid", "content": meta.identifier})
    ET.SubElement(head, f"{{{SMIL_NS}}}meta", {"name": "dtb:generator", "content": "DAISY Generator"})
    
    # Approximate duration from the last clip end.
    chapter_duration = max((s.end or 0.0) for s in chapter.sentences)
    ET.SubElement(head, f"{{{SMIL_NS}}}meta", {"name": "dtb:totalElapsedTime", "content": format_smil_time(chapter_duration)})

    body = ET.SubElement(root, f"{{{SMIL_NS}}}body")
    
    # Calculate total duration for seq
    total_duration = max((s.end or 0.0) for s in chapter.sentences) if chapter.sentences else 0.0
    
    # SMIL files live in smil/, so hop one level up to reach main.xml and media/.
    main_ref = (Path("..") / main_href).as_posix()
    textref = f"{main_ref}#{chapter.cid}"
    
    seq_attrs = {
        "id": f"seq_{chapter.cid}",
        "dur": format_smil_time(total_duration),
        "fill": "remove",
        # Help readers map this SMIL to the DTBook anchor for the chapter.
        "textref": textref,
    }
    seq = ET.SubElement(body, f"{{{SMIL_NS}}}seq", seq_attrs)

    for sentence in chapter.sentences:
        audio_src = sentence.audio_path or chapter.audio_path
        if audio_src is None:
            raise ValueError(f"No audio source found for sentence {sentence.sid}")
        audio_href = (Path("..") / audio_src).as_posix()

        par = ET.SubElement(seq, f"{{{SMIL_NS}}}par", {"id": f"par_{sentence.sid}"})
        ET.SubElement(par, f"{{{SMIL_NS}}}text", {"id": f"text_{sentence.sid}", "src": f"{main_ref}#{sentence.sid}"})
        audio_attrs = {"src": audio_href}
        if sentence.start is not None:
            audio_attrs["clipBegin"] = format_smil_time(sentence.start)
        if sentence.end is not None:
            audio_attrs["clipEnd"] = format_smil_time(sentence.end)
        ET.SubElement(par, f"{{{SMIL_NS}}}audio", audio_attrs)

    tree = ET.ElementTree(root)
    indent(tree)
    doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE smil\n'
        '  PUBLIC "-//NISO//DTD dtbsmil 2005-2//EN" "http://www.daisy.org/z3986/2005/dtbsmil-2005-2.dtd">\n'
    )
    with out_path.open("wb") as f:
        f.write(doctype.encode("utf-8"))
        tree.write(f, encoding="utf-8", xml_declaration=False)


def build_opf(meta: Metadata, chapters: List[Chapter], main_href: str, ncx_href: str, out_path: Path, cover_href: Optional[str]) -> None:
    ET.register_namespace("", OPF_NS)
    ET.register_namespace("dc", DC_NS)
    package = ET.Element(
        f"{{{OPF_NS}}}package",
        {"unique-identifier": "uid", "version": "2005-1"},
    )

    metadata = ET.SubElement(package, f"{{{OPF_NS}}}metadata")
    
    # Create nested dc-metadata structure
    dc_metadata = ET.SubElement(metadata, "dc-metadata", {
        "xmlns:dc": DC_NS,
        "xmlns:oebpackage": OPF_NS
    })
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Format").text = "ANSI/NISO Z39.86-2005"
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Language").text = meta.language
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Date").text = meta.date
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Creator").text = meta.creator
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Publisher").text = meta.publisher
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Title").text = meta.title
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Subject").text = meta.subject
    ET.SubElement(dc_metadata, f"{{{DC_NS}}}Identifier").text = meta.identifier
    uid_elem = ET.SubElement(dc_metadata, f"{{{DC_NS}}}Identifier", {"id": "uid"})
    uid_elem.text = meta.identifier
    
    # Create x-metadata structure
    x_metadata = ET.SubElement(metadata, "x-metadata")
    ET.SubElement(x_metadata, "meta", {"name": "dtb:multimediaType", "content": "audioFullText"})
    
    # Calculate total time from all chapters
    total_time = 0.0
    for chapter in chapters:
        if chapter.sentences:
            total_time += max((s.end or 0.0) for s in chapter.sentences)
    ET.SubElement(x_metadata, "meta", {"name": "dtb:totalTime", "content": format_smil_time(total_time)})
    ET.SubElement(x_metadata, "meta", {"name": "dtb:multimediaContent", "content": "audio,text"})
    
    # Add cover reference in x-metadata if cover exists
    if cover_href:
        ET.SubElement(x_metadata, "meta", {"name": "cover", "content": "cover"})

    manifest = ET.SubElement(package, f"{{{OPF_NS}}}manifest")
    ET.SubElement(manifest, f"{{{OPF_NS}}}item", {"id": "dtbook", "href": main_href, "media-type": "application/x-dtbook+xml"})
    ET.SubElement(manifest, f"{{{OPF_NS}}}item", {"id": "ncx", "href": ncx_href, "media-type": "application/x-dtbncx+xml"})

    for chapter in chapters:
        ET.SubElement(manifest, f"{{{OPF_NS}}}item", {"id": f"smil-{chapter.cid[1:]}", "href": f"smil/{chapter.smil_name}", "media-type": "application/smil"})

        sentence_audio_used = any(sentence.audio_path for sentence in chapter.sentences)
        audio_seen: set[str] = set()
        audio_counter = 1
        if sentence_audio_used:
            audio_sources = [
                sentence.audio_path or chapter.audio_path
                for sentence in chapter.sentences
            ]
        else:
            audio_sources = [chapter.audio_path]

        for audio_path in audio_sources:
            if audio_path is None:
                continue
            href = audio_path.as_posix()
            if href in audio_seen:
                continue
            audio_seen.add(href)

            audio_id = f"audio_{chapter.cid}_{audio_counter}" if sentence_audio_used else f"audio_{chapter.cid}"
            audio_counter += 1
            ET.SubElement(
                manifest,
                f"{{{OPF_NS}}}item",
                {"id": audio_id, "href": href, "media-type": audio_media_type(audio_path)},
            )
    if cover_href:
        ET.SubElement(manifest, f"{{{OPF_NS}}}item", {"id": "cover", "href": cover_href, "media-type": "image/jpeg"})

    spine = ET.SubElement(package, f"{{{OPF_NS}}}spine", {"toc": "ncx"})
    # Spine should only contain SMIL files for proper audio playback
    for chapter in chapters:
        ET.SubElement(spine, f"{{{OPF_NS}}}itemref", {"idref": f"smil-{chapter.cid[1:]}"})

    tree = ET.ElementTree(package)
    indent(tree)
    doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE package\n'
        '  PUBLIC "+//ISBN 0-9673008-1-9//DTD OEB 1.2 Package//EN" "http://openebook.org/dtds/oeb-1.2/oebpkg12.dtd">\n'
    )
    with out_path.open("wb") as f:
        f.write(doctype.encode("utf-8"))
        tree.write(f, encoding="utf-8", xml_declaration=False)


def build_ncx(
    meta: Metadata,
    chapters: List[Chapter],
    main_href: str,
    out_path: Path,
    include_sentence_nav: bool,
) -> None:
    ET.register_namespace("", NCX_NS)
    root = ET.Element(f"{{{NCX_NS}}}ncx", {"version": "2005-1", "xml:lang": meta.language})
    head = ET.SubElement(root, f"{{{NCX_NS}}}head")
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:uid", "content": meta.identifier})
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:generator", "content": "DAISY Generator"})
    # Required DAISY 3 metadata for audio playback.
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:multimediaType", "content": "audioFullText"})
    total_elapsed = sum(max((s.end or 0.0) for s in chapter.sentences) for chapter in chapters)
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:totalElapsedTime", "content": format_smil_time(total_elapsed)})
    # Always reflect sentence-level depth when requested so readers don't collapse the tree.
    ET.SubElement(
        head,
        f"{{{NCX_NS}}}meta",
        {"name": "dtb:depth", "content": "2" if include_sentence_nav else "1"},
    )
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:totalPageCount", "content": "0"})
    ET.SubElement(head, f"{{{NCX_NS}}}meta", {"name": "dtb:maxPageNumber", "content": "0"})

    doc_title = ET.SubElement(root, f"{{{NCX_NS}}}docTitle")
    ET.SubElement(doc_title, f"{{{NCX_NS}}}text").text = meta.title

    nav_map = ET.SubElement(root, f"{{{NCX_NS}}}navMap")
    play_order = 1
    for chapter in chapters:
        nav_point = ET.SubElement(nav_map, f"{{{NCX_NS}}}navPoint", {"id": f"np_{chapter.cid}", "playOrder": str(play_order)})
        play_order += 1
        nav_label = ET.SubElement(nav_point, f"{{{NCX_NS}}}navLabel")
        ET.SubElement(nav_label, f"{{{NCX_NS}}}text").text = chapter.title

        # Point nav to SMIL so playback starts with audio while text links live in SMIL <text> refs.
        ET.SubElement(nav_point, f"{{{NCX_NS}}}content", {"src": f"smil/{chapter.smil_name}#seq_{chapter.cid}"})

        # Only add sentence-level nav if explicitly requested (default is chapter-only for better navigation)
        if include_sentence_nav:
            for sentence in chapter.sentences:
                child_np = ET.SubElement(nav_point, f"{{{NCX_NS}}}navPoint", {"id": f"np_{sentence.sid}", "playOrder": str(play_order)})
                play_order += 1
                child_label = ET.SubElement(child_np, f"{{{NCX_NS}}}navLabel")
                # Truncate long sentences in nav for readability
                nav_text = sentence.text[:100] + "..." if len(sentence.text) > 100 else sentence.text
                ET.SubElement(child_label, f"{{{NCX_NS}}}text").text = nav_text
                ET.SubElement(child_np, f"{{{NCX_NS}}}content", {"src": f"smil/{chapter.smil_name}#par_{sentence.sid}"})

    tree = ET.ElementTree(root)
    indent(tree)
    doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE ncx\n'
        '  PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">\n'
    )
    with out_path.open("wb") as f:
        f.write(doctype.encode("utf-8"))
        tree.write(f, encoding="utf-8", xml_declaration=False)


def chapter_sort_key(path: Path) -> tuple[str, int, str]:
    """Sort chapters by base name then numeric part (e.g., stem, stem_P2)."""
    stem = path.name.replace("_word_level_matches.json", "")
    base = stem
    part = 1
    m = re.match(r"^(.*?)(?:_P?(\d+))$", stem, flags=re.IGNORECASE)
    if m and m.group(2):
        base = m.group(1)
        part = int(m.group(2))
    return (base.lower(), part, stem.lower())


def collect_chapters(json_dir: Path, audio_dir: Path) -> List[Chapter]:
    audio_map = discover_audio_map(audio_dir)

    json_files = sorted(json_dir.glob("*_word_level_matches.json"), key=chapter_sort_key)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    chapters: List[Chapter] = []
    for idx, json_path in enumerate(json_files, start=1):
        stem = json_path.name.replace("_word_level_matches.json", "")
        audio_path = stem_to_audio(stem, audio_map)
        sentences = load_sentences(json_path, idx, audio_path)
        title = humanize_chapter_title(stem)
        cid = f"c{idx:02d}"
        smil_name = f"{stem}.smil"
        chapters.append(Chapter(json_path, audio_path, title, cid, sentences, smil_name))
    return chapters


def attach_sentence_audio(
    chapters: List[Chapter],
    sentence_audio_dir: Path,
    out_dir: Path,
    media_dir: Path,
    copy_media: bool,
) -> bool:
    """
    When sentence-level WAVs exist (audio_segments_method_w/<stem>/sentence_00001.wav),
    attach them to Sentence objects so SMIL can reference trimmed audio directly.
    Returns True if at least one segment was wired up.
    """
    found_any = False
    for chapter in chapters:
        stem = chapter.smil_name.replace(".smil", "")
        seg_dir = sentence_audio_dir / stem
        if not seg_dir.exists():
            continue

        for sentence in chapter.sentences:
            seg_name = f"sentence_{sentence.line_number:05d}.wav"
            seg_src = seg_dir / seg_name
            if not seg_src.exists():
                print(
                    f"Missing segment for {stem} line {sentence.line_number}: {seg_src}",
                    file=sys.stderr,
                )
                continue

            dest = media_dir / "segments" / stem / seg_name
            packaged_path = maybe_copy(seg_src, dest, copy_media)
            sentence.audio_path = Path(os.path.relpath(packaged_path, out_dir))
            # When using pre-cut audio, let SMIL play the whole file (no clip attrs).
            sentence.start = None
            sentence.end = None
            found_any = True

    return found_any


def maybe_copy(src: Path, dest: Path, enabled: bool) -> Path:
    """Copy file if enabled; otherwise return original path."""
    if not enabled:
        return src
    ensure_dir(dest.parent)
    shutil.copyfile(src, dest)
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate DAISY 3 package files from JSON + MP3 inputs.")
    parser.add_argument("--json-dir", default="output_xaxoi", type=Path, help="Directory containing *_word_level_matches.json files.")
    parser.add_argument("--audio-dir", default=Path("data/Audio-XaXoiThonNguaGia"), type=Path, help="Directory containing MP3s.")
    parser.add_argument("--sentence-audio-dir", default=Path("output_xaxoi/audio_segments_method_w"), type=Path, help="Directory containing per-sentence WAV segments (subfolders per chapter).")
    parser.add_argument("--cover", default=Path("data/xa-xoi-thon-ngua-gia-rs.jpg"), type=Path, help="Cover image (JPEG).")
    parser.add_argument("--out-dir", default=Path("build/daisy"), type=Path, help="Output directory for DAISY package.")
    parser.add_argument("--no-copy-media", action="store_true", help="Do not copy audio/cover into the output; reference existing paths.")
    parser.add_argument("--use-sentence-audio", action="store_true", help="Use per-sentence WAV segments; otherwise use chapter-level audio timings.")
    parser.add_argument("--include-sentence-nav", action="store_true", help="Add navPoints for every sentence in navigation.ncx (default: only chapters for cleaner navigation).")

    parser.add_argument("--title", default="Xa Xôi Thôn Ngựa Già")
    parser.add_argument("--creator", default="Ma Văn Kháng")
    parser.add_argument("--subject", default="Văn học & Tiểu thuyết")
    parser.add_argument("--description", default=(
        "Tập truyện vừa Xa xôi thôn Ngựa Già của nhà văn Ma Văn Kháng một lần nữa "
        "cho người đọc thấy cái nhìn đa chiều về những mặt tích cực cũng như tiêu cực của "
        "đời sống xã hội. Cuốn sách gồm sáu truyện vừa về đời sống văn hóa tinh thần của "
        "người dân những bản làng vùng núi phía Bắc."
    ))
    parser.add_argument("--publisher", default="NXB Phụ Nữ")
    parser.add_argument("--date", default="2013")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--identifier", default="9786045617953", help="Primary identifier (also used as dtb:uid).")
    parser.add_argument("--source-isbn", default="9786045617953")
    parser.add_argument("--source-url", default="https://thuviensachnoihuongduong.com")
    parser.add_argument("--note", default="Tập truyện vừa gồm 6 truyện. Người đọc: Như Minh. Thư viện Sách nói Hướng Dương.")
    parser.add_argument("--reader", default="Như Minh")
    parser.add_argument("--collector", default="Thư viện Sách nói Hướng Dương")

    args = parser.parse_args(argv)

    copy_media = not args.no_copy_media
    out_dir: Path = args.out_dir
    smil_dir = out_dir / "smil"
    media_dir = out_dir / "media"
    ensure_dir(out_dir)
    ensure_dir(smil_dir)
    if copy_media:
        ensure_dir(media_dir)

    use_sentence_audio = (
        args.use_sentence_audio
        and args.sentence_audio_dir is not None
        and args.sentence_audio_dir.exists()
    )
    if args.use_sentence_audio and args.sentence_audio_dir and not args.sentence_audio_dir.exists():
        print(f"Sentence audio dir not found: {args.sentence_audio_dir} (falling back to chapter audio)", file=sys.stderr)

    meta = Metadata(
        title=args.title,
        creator=args.creator,
        subject=args.subject,
        description=args.description,
        publisher=args.publisher,
        date=args.date,
        language=args.language,
        identifier=args.identifier,
        source_isbn=args.source_isbn,
        source_url=args.source_url,
        note=args.note,
        reader=args.reader,
        collector=args.collector,
    )

    chapters = collect_chapters(args.json_dir, args.audio_dir)

    # Copy audio and rewrite chapter.audio_path to be package-relative.
    for chapter in chapters:
        dest = media_dir / chapter.audio_path.name
        packaged_path = maybe_copy(chapter.audio_path, dest, copy_media)
        # Store href relative to out_dir for manifest/SMIL references.
        chapter.audio_path = Path(os.path.relpath(packaged_path, out_dir))

    sentence_audio_used = False
    if use_sentence_audio:
        sentence_audio_used = attach_sentence_audio(
            chapters,
            args.sentence_audio_dir,
            out_dir,
            media_dir,
            copy_media,
        )
        if sentence_audio_used:
            print(f"Using sentence-level audio from {args.sentence_audio_dir}")
        else:
            print(f"No sentence-level WAVs found under {args.sentence_audio_dir}; using chapter audio timings.")
    else:
        print("Sentence-level audio disabled or unavailable; using chapter audio with clip timings.")

    cover_href = None
    if args.cover and args.cover.exists():
        cover_dest = media_dir / args.cover.name
        cover_path = maybe_copy(args.cover, cover_dest, copy_media)
        cover_href = Path(os.path.relpath(cover_path, out_dir)).as_posix()

    main_xml = out_dir / "main.xml"
    build_dtbook(meta, chapters, main_xml)

    # Build SMIL files.
    for chapter in chapters:
        smil_path = smil_dir / chapter.smil_name
        build_smil(chapter, "main.xml", smil_path, meta)

    ncx_path = out_dir / "navigation.ncx"
    print(f"Building NCX -> {ncx_path} (include_sentence_nav={args.include_sentence_nav})")
    build_ncx(meta, chapters, "main.xml", ncx_path, include_sentence_nav=args.include_sentence_nav)

    opf_path = out_dir / "main.opf"
    build_opf(meta, chapters, "main.xml", "navigation.ncx", opf_path, cover_href)

    print(f"Wrote DTBook to {main_xml}")
    print(f"Wrote {len(chapters)} SMIL files to {smil_dir}")
    print(f"Wrote OPF to {opf_path}")
    print(f"Wrote NCX to {ncx_path}")
    if copy_media:
        print(f"Copied media to {media_dir}")
    else:
        print("Media copying disabled; manifest references existing files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

