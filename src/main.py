#!/usr/bin/env python3
"""
Main script to run sentence splitting on all text files in the data directory.
"""

import os
import sys
from sentence_splitter import process_text


def count_sentences(file_path):
    """
    Count the number of sentences in a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Number of sentences, or None if error occurred
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = process_text(text)
        return len(sentences)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        return None


def count_sentences_in_data_files():
    """
    Count sentences in the 2 original text files in the data directory.
    """
    # Get the data directory path (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Files to count
    files_to_count = [
        'Thiên thần nhỏ của tôi - Nguyễn Nhật Ánh.txt',
        'Xa Xôi Thôn Ngựa Già.txt'
    ]
    
    print("Counting sentences in data files...")
    print(f"Data directory: {data_dir}\n")
    
    total_sentences = 0
    for filename in files_to_count:
        file_path = os.path.join(data_dir, filename)
        count = count_sentences(file_path)
        if count is not None:
            print(f"  {filename}: {count} sentences")
            total_sentences += count
        else:
            print(f"  {filename}: Error counting sentences")
    
    print(f"\nTotal sentences: {total_sentences}")
    return total_sentences


def process_file(input_path, output_path=None):
    """
    Process a single file with sentence splitting.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (optional)
    """
    # Determine output file if not specified
    if output_path is None:
        if '.' in input_path:
            parts = input_path.rsplit('.', 1)
            output_path = f"{parts[0]}.processed.{parts[1]}"
        else:
            output_path = f"{input_path}.processed"
    
    # Read input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading file '{input_path}': {e}", file=sys.stderr)
        return False
    
    # Process text
    sentences = process_text(text)
    
    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"✓ Processed '{input_path}': {len(sentences)} sentences → '{output_path}'")
        return True
    except Exception as e:
        print(f"Error writing file '{output_path}': {e}", file=sys.stderr)
        return False


def main():
    """Main function to process all files in the data directory."""
    # Get the data directory path (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Files to process
    files_to_process = [
        'Thiên thần nhỏ của tôi - Nguyễn Nhật Ánh.txt',
        'Xa Xôi Thôn Ngựa Già.txt'
    ]
    
    print("Starting sentence splitting process...")
    print(f"Data directory: {data_dir}\n")
    
    success_count = 0
    for filename in files_to_process:
        input_path = os.path.join(data_dir, filename)
        if os.path.exists(input_path):
            if process_file(input_path):
                success_count += 1
            print()  # Empty line for readability
        else:
            print(f"✗ File not found: '{input_path}'", file=sys.stderr)
            print()
    
    print(f"Completed: {success_count}/{len(files_to_process)} files processed successfully.")


if __name__ == '__main__':
    main()

