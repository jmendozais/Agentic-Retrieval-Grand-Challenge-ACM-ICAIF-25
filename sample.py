#!/usr/bin/env python3
"""
subsample_jsonl.py

Provides a function to subsample lines from a .jsonl file and save them to an output file.
Prints total number of lines and percentage subsampled.

Usage:
    from subsample_jsonl import subsample_jsonl
    subsample_jsonl("data.jsonl", 100, "sample.jsonl")

Or run as a script:
    python subsample_jsonl.py input.jsonl 100 output.jsonl
"""

import sys
import random
import argparse
import os

import dataset

def subsample_jsonl(input_path: str, sample_size: int, output_path: str) -> None:
    """
    Subsample up to `sample_size` lines from a JSONL file at `input_path` and write them
    to `output_path`. Prints the total number of lines and the percentage subsampled.
    Sampling is done uniformly without replacement using reservoir sampling (one-pass,
    memory bounded).

    Args:
        input_path: Path to the input .jsonl file.
        sample_size: Number of lines to subsample (must be >= 1).
        output_path: Path where the subsampled .jsonl will be written.

    Raises:
        ValueError: if sample_size < 1.
        IOError: if input file cannot be read or output cannot be written.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    reservoir = []
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                # replace elements with decreasing probability
                j = random.randint(0, total - 1)
                if j < sample_size:
                    reservoir[j] = line

    # If file was empty
    if total == 0:
        print("Total lines: 0")
        print("Subsampled: 0 lines (0.00%)")
        # create empty output file
        open(output_path, "w", encoding="utf-8").close()
        return

    # If requested sample_size > total, reservoir length will be total
    selected = len(reservoir)
    pct = (selected / total) * 100.0

    # Write selected lines to output file
    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(reservoir)

    print(f"Total lines: {total}")
    print(f"Subsampled: {selected} lines ({pct:.2f}%)")

def subsample_dataset(input_path, 
                      doc_dev_size=100, 
                      doc_test_size=100,
                      chunk_dev_size=100,
                      chunk_test_size=100,
                      output_path=None):
    
    
    if output_path is None:
        output_path = input_path + f"_sampled_{doc_dev_size}_{doc_test_size}_{chunk_dev_size}_{chunk_test_size}"
    
    os.makedirs(output_path, exist_ok=True)

    subsample_jsonl(
        os.path.join(input_path, dataset.DOC_DEV_FNAME), 
        doc_dev_size,
        os.path.join(output_path, dataset.DOC_DEV_FNAME)
    )

    subsample_jsonl(
        os.path.join(input_path, dataset.DOC_EVAL_FNAME), 
        doc_test_size,
        os.path.join(output_path, dataset.DOC_EVAL_FNAME)
    )

    subsample_jsonl(
        os.path.join(input_path, dataset.CHUNK_DEV_FNAME), 
        chunk_dev_size,
        os.path.join(output_path, dataset.CHUNK_DEV_FNAME)
    )

    subsample_jsonl(
        os.path.join(input_path, dataset.CHUNK_EVAL_FNAME), 
        chunk_test_size,
        os.path.join(output_path, dataset.CHUNK_EVAL_FNAME)
    )
    

if __name__ == "__main__":
    input_file = "/home/julio/Dataset/ACM-ICAIF-25/acm-icaif-25-ai-agentic-retrieval-grand-challenge"
    output_file = "test.jsonl"

    subsample_dataset(input_file, 2, 2, 2, 2)
