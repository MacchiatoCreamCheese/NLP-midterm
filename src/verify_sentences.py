#!/usr/bin/env python3
"""
Function to verify that all sentences from text files in the data directory
have been split and matched in the output folder.
"""

import os
import sys
from pathlib import Path
from sentence_splitter import process_text
from typing import List, Tuple, Dict


def get_all_text_files(data_dir: str) -> List[Path]:
    """
    Get all text files from Text-* folders in the data directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of Path objects to text files
    """
    data_path = Path(data_dir)
    text_files = []
    
    # Find all Text-* folders
    for text_folder in data_path.glob("Text-*"):
        # Get all .txt files in each Text-* folder
        text_files.extend(text_folder.glob("*.txt"))
    
    return sorted(text_files)


def load_sentences_from_text_file(file_path: Path) -> List[str]:
    """
    Load and split sentences from a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of sentences (normalized)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = process_text(text)
        # Normalize: strip whitespace for comparison
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        return []


def load_sentences_from_output(output_dir: str) -> List[str]:
    """
    Load all sentences from the output folder.
    
    Args:
        output_dir: Path to the output folder
        
    Returns:
        List of sentences (normalized), sorted by file index
    """
    output_path = Path(output_dir)
    sentences = []
    
    # Get all sentence_*.txt files and sort them by number
    sentence_files = sorted(
        output_path.glob("sentence_*.txt"),
        key=lambda x: int(x.stem.split('_')[1])
    )
    
    for file_path in sentence_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    sentences.append(content)
        except Exception as e:
            print(f"Error reading output file '{file_path}': {e}", file=sys.stderr)
    
    return sentences


def normalize_sentence(sentence: str) -> str:
    """
    Normalize a sentence for comparison.
    - Strip whitespace
    - Normalize spaces
    
    Args:
        sentence: Input sentence
        
    Returns:
        Normalized sentence
    """
    import re
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', sentence.strip())
    return normalized


def check_file_sentences_match(
    text_file: Path,
    output_sentences: List[str],
    normalized_output_set: set
) -> Dict:
    """
    Check if all sentences from a single text file match the output.
    
    Args:
        text_file: Path to the text file to check
        output_sentences: List of all output sentences
        normalized_output_set: Set of normalized output sentences for fast lookup
        
    Returns:
        Dictionary with verification results for this file
    """
    # Load sentences from text file
    source_sentences = load_sentences_from_text_file(text_file)
    
    # Normalize sentences
    normalized_source = [normalize_sentence(s) for s in source_sentences]
    source_set = set(normalized_source)
    
    # Find missing and extra sentences
    missing_sentences = []
    for i, sent in enumerate(normalized_source):
        if sent not in normalized_output_set:
            missing_sentences.append((i, source_sentences[i]))
    
    # Check which output sentences match this file
    matching_output = [sent for sent in normalized_source if sent in normalized_output_set]
    
    matches = len(matching_output)
    all_match = len(missing_sentences) == 0 and matches == len(normalized_source)
    
    return {
        'file_path': str(text_file),
        'file_name': text_file.name,
        'total_sentences': len(normalized_source),
        'matches': matches,
        'missing_count': len(missing_sentences),
        'missing_sentences': missing_sentences,
        'match_percentage': (matches / len(normalized_source) * 100) if normalized_source else 0,
        'all_match': all_match
    }


def check_sentences_match(
    data_dir: str = None,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Check if all sentences from text files in the data directory
    are present and match in the output folder.
    Checks each file individually.
    
    Args:
        data_dir: Path to data directory (default: ../data relative to script)
        output_dir: Path to output directory (default: ../output_whisper_matched relative to script)
        verbose: If True, print detailed information
        
    Returns:
        Tuple of (all_match: bool, report: dict)
        Report contains:
            - file_results: List of results for each file
            - total_source_sentences: Total sentences in all source files
            - total_output_sentences: Total sentences in output folder
            - files_with_all_matches: Number of files where all sentences match
            - files_checked: Number of files checked
    """
    # Set default paths
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'output_whisper_matched')
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Get all text files
    text_files = get_all_text_files(data_dir)
    
    if verbose:
        print(f"Found {len(text_files)} text file(s) in data directory")
        print(f"Output directory: {output_dir}\n")
    
    # Load all sentences from output folder
    if verbose:
        print("Loading sentences from output folder...")
    output_sentences = load_sentences_from_output(output_dir)
    normalized_output = [normalize_sentence(s) for s in output_sentences]
    normalized_output_set = set(normalized_output)
    
    if verbose:
        print(f"Total output sentences: {len(output_sentences)}\n")
    
    # Check each file individually
    file_results = []
    total_source_sentences = 0
    files_with_all_matches = 0
    
    for text_file in text_files:
        if verbose:
            print(f"{'='*60}")
            print(f"Checking: {text_file.name}")
            print(f"{'='*60}")
        
        file_result = check_file_sentences_match(
            text_file,
            output_sentences,
            normalized_output_set
        )
        
        file_results.append(file_result)
        total_source_sentences += file_result['total_sentences']
        
        if file_result['all_match']:
            files_with_all_matches += 1
        
        if verbose:
            print(f"  Total sentences: {file_result['total_sentences']}")
            print(f"  Matches: {file_result['matches']}")
            print(f"  Missing: {file_result['missing_count']}")
            print(f"  Match percentage: {file_result['match_percentage']:.2f}%")
            print(f"  Status: {'✓ ALL MATCH' if file_result['all_match'] else '✗ SOME MISSING'}")
            
            if file_result['missing_sentences']:
                print(f"\n  Missing sentences (first 10):")
                for idx, (orig_idx, sent) in enumerate(file_result['missing_sentences'][:10], 1):
                    print(f"    {idx}. [{orig_idx}] {sent[:80]}{'...' if len(sent) > 80 else ''}")
                if len(file_result['missing_sentences']) > 10:
                    print(f"    ... and {len(file_result['missing_sentences']) - 10} more")
            print()
    
    all_match = files_with_all_matches == len(text_files)
    
    report = {
        'file_results': file_results,
        'total_source_sentences': total_source_sentences,
        'total_output_sentences': len(output_sentences),
        'files_with_all_matches': files_with_all_matches,
        'files_checked': len(text_files),
        'all_match': all_match
    }
    
    # Print summary
    if verbose:
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Files checked: {len(text_files)}")
        print(f"Files with all matches: {files_with_all_matches}")
        print(f"Total source sentences: {total_source_sentences}")
        print(f"Total output sentences: {len(output_sentences)}")
        print(f"\nAll files match: {all_match}")
        
        if not all_match:
            print(f"\nFiles with missing sentences:")
            for file_result in file_results:
                if not file_result['all_match']:
                    print(f"  - {file_result['file_name']}: {file_result['missing_count']} missing")
    
    return all_match, report


def main():
    """Main function to run the verification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify that all sentences from text files are present in output folder'
    )
    parser.add_argument(
        '--data-dir',
        help='Path to data directory (default: ../data)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='Path to output directory (default: ../output_whisper_matched)',
        default=None
    )
    parser.add_argument(
        '-q', '--quiet',
        help='Suppress detailed output',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    try:
        all_match, report = check_sentences_match(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        # Exit with appropriate code
        sys.exit(0 if all_match else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

