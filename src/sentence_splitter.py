#!/usr/bin/env python3
"""
Script to reformat text files so that each line contains one sentence.
- Removes all existing newlines
- Inserts newlines after sentence-ending punctuation: . : ! ?
- Preserves "..." as a single unit (doesn't split on each dot)
"""

import re
import sys
import argparse


def process_text(text):
    """
    Process text to split sentences while preserving "..."
    
    Args:
        text: Input text string
        
    Returns:
        List of sentences
    """
    # Remove all existing newlines and replace with spaces
    text = re.sub(r'\n+', ' ', text)
    # Also remove carriage returns
    text = re.sub(r'\r+', ' ', text)
    # Remove extra whitespace but keep single spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Strategy: Replace "..." with a placeholder to protect it
    placeholder = "___ELLIPSIS___"
    text = text.replace("...", placeholder)
    
    # Split on sentence-ending punctuation: . : ! ?
    # Pattern: look for punctuation followed by space or end of string
    # Use positive lookahead to keep the punctuation with the sentence
    pattern = r'([.:!?])(?=\s|$)'
    
    # Split the text, keeping the delimiters
    parts = re.split(pattern, text)
    
    # Reconstruct sentences
    result = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ['.', ':', '!', '?']:
            # This is a sentence ending with punctuation
            sentence = parts[i] + parts[i + 1]
            sentence = sentence.strip()
            if sentence:
                # Restore ellipsis
                sentence = sentence.replace(placeholder, "...")
                result.append(sentence)
            i += 2
        else:
            # Trailing text without punctuation or empty
            if parts[i].strip():
                sentence = parts[i].strip()
                # Restore ellipsis
                sentence = sentence.replace(placeholder, "...")
                if sentence:
                    result.append(sentence)
            i += 1
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Reformat text file so each line is a sentence'
    )
    parser.add_argument(
        'input_file',
        help='Input text file to process'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: overwrites input file with .processed suffix)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process text
    sentences = process_text(text)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Add .processed before extension
        if '.' in args.input_file:
            parts = args.input_file.rsplit('.', 1)
            output_file = f"{parts[0]}.processed.{parts[1]}"
        else:
            output_file = f"{args.input_file}.processed"
    
    # Write output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"Processed {len(sentences)} sentences. Output written to: {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

