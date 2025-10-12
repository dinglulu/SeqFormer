#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeqFormer Consensus Prediction CLI

Input FASTA format with quality lines:
>seq1
AGCTGACTGACGT
IIIIIIIIIIIIII
>seq2
AGCTGACTAAGCT
IIIIIIIIIIIIII
"""

import os
import json
import numpy as np
import torch
from p_tqdm import p_map
from argparse import ArgumentParser
import evaluate_v2


# ---------------------------------------------------------
# Built-in model configurations
# ---------------------------------------------------------
MODEL_CONFIG = {
    "ngs": {
        "model_path": "./model/Illumina.pth",
        "qual_vocab_size": 41,   # Illumina model
        "qchar": "I"
    },
    "tgs": {
        "model_path": "./model/ONT.pth",
        "qual_vocab_size": 91,   # Nanopore model
        "qchar": "Z"
    }
}


# ---------------------------------------------------------
# Read FASTA file with quality lines
# ---------------------------------------------------------
def read_fasta_with_qual(file_path, sequencing_type="ngs"):
    """
    Read FASTA or FASTA+QUAL formatted file.
    Automatically detects whether quality lines exist.
    If a quality line starts with '>', replace it with the next ASCII character (e.g., '>' -> '?').
    """

    lines = [line.strip() for line in open(file_path, "r") if line.strip()]
    reads, quals = [], []

    if not lines[0].startswith(">"):
        raise ValueError(f"❌ Invalid FASTA format: first line must start with '>' in {file_path}")

    # Detect if there are quality lines
    has_quality = False
    for i in range(1, len(lines)):
        if not lines[i].startswith(">"):
            if i + 1 < len(lines) and not lines[i + 1].startswith(">"):
                has_quality = True
            break

    qchar = MODEL_CONFIG[sequencing_type]["qchar"]
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(">") and len(line) > 1 and line[1].isalnum():
            if i + 1 >= len(lines):
                raise ValueError(f"❌ Missing sequence line after header at line {i+1} in {file_path}")

            seq = lines[i + 1]
            if not all(c in "ACGTNacgtn" for c in seq):
                raise ValueError(f"❌ Invalid base in sequence near {line} in {file_path}")

            # Case 1: with quality (sequence + quality)
            if has_quality and i + 2 < len(lines) and not lines[i + 2].startswith(">"):
                qual = lines[i + 2]
                # 修复质量值以 '>' 开头的情况
                if qual.startswith(">"):
                    fixed_first_char = chr(ord(">") + 1)  # '>' -> '?'
                    qual = fixed_first_char + qual[1:]
                reads.append(seq)
                quals.append(qual)
                i += 3  # header, seq, qual
            else:
                # Case 2: no quality line (auto-fill)
                reads.append(seq)
                quals.append(qchar * len(seq))
                i += 2
        else:
            i += 1

    if len(reads) == 0:
        raise ValueError(f"❌ No valid sequence entries found in {file_path}")

    print(f"✅ Detected format: {'FASTA+QUAL' if has_quality else 'FASTA-only'}")
    print(f"✅ Loaded {len(reads)} reads from {file_path} ({sequencing_type.upper()})")

    ref = reads[0]
    return {
        "cluster_0": {
            "refs": ref,
            "seqs": reads,
            "quals": quals
        }
    }



# ---------------------------------------------------------
# Simple majority-vote consensus
# ---------------------------------------------------------
def vote_consensus(msa):
    arr = np.array([list(seq) for seq in msa])
    consensus = []
    for col in arr.T:
        unique, counts = np.unique(col, return_counts=True)
        consensus.append(unique[np.argmax(counts)])
    return "".join(consensus).replace("-", "")


# ---------------------------------------------------------
# Run BSAlign and generate consensus
# ---------------------------------------------------------
def bsalign_process_v2(seqs, file_dir, file_idx):
    bsalign_exe_path = "./dna_storage_bench_tools/Reconstruction/bsalign/bsalign"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    fasta_path = os.path.join(file_dir, f"{file_idx}.fasta")
    consensus_path = os.path.join(file_dir, f"{file_idx}.consensus")
    alignment_path = os.path.join(file_dir, f"{file_idx}.alignment")

    with open(fasta_path, "w") as f:
        for s_id, s in enumerate(seqs):
            f.write(f">{s_id}\n{s}\n")

    _ = os.system(f"{bsalign_exe_path} poa {fasta_path} -o {consensus_path} -L > {alignment_path}")

    with open(alignment_path) as f:
        msa = [
            line.strip().split(" ")[3].upper().replace(".", "-")
            for line in f.readlines()[2: 2 + len(seqs)]
        ]

    cns = vote_consensus(msa)

    os.remove(fasta_path)
    os.remove(consensus_path)
    os.remove(alignment_path)
    return file_idx, msa, cns


def bsalign_process_parallel(data, file_dir, num_cpus=4):
    data_list = [(item["seqs"], file_dir, cid) for cid, item in data.items()]
    results = p_map(lambda kv: bsalign_process_v2(kv[0], kv[1], kv[2]), data_list, num_cpus=num_cpus)
    return dict([(k, msa) for k, msa, cns in results]), dict([(k, cns) for k, msa, cns in results])


# ---------------------------------------------------------
# Main consensus prediction pipeline
# ---------------------------------------------------------
def run_consensus_prediction(args):
    # Determine sequencing type
    if args.ngs and args.tgs:
        raise ValueError("❌ Invalid arguments: cannot specify both --ngs and --tgs.")
    if not args.ngs and not args.tgs:
        raise ValueError("❌ Please specify either --ngs (Illumina) or --tgs (Nanopore).")

    sequencing_type = "tgs" if args.tgs else "ngs"

    # Load model config
    model_path = MODEL_CONFIG[sequencing_type]["model_path"]
    qual_vocab_size = MODEL_CONFIG[sequencing_type]["qual_vocab_size"]

    print(f"🔧 Sequencing type: {sequencing_type.upper()}")
    print(f"🔧 Model path: {model_path}")
    print(f"🔧 Qual vocab size: {qual_vocab_size}")

    batched_data = read_fasta_with_qual(args.input, sequencing_type)

    predictor = evaluate_v2.ConsensusPredictor(
        model_path=model_path,
        qual_vocab_size=qual_vocab_size,
        context_len=args.context_len,
        max_depth=args.max_depth,
        batch_size=args.batch_size,
        gpu=args.gpu,
    )

    bs_msa_all, bs_cns_all = bsalign_process_parallel(batched_data, args.bsalign_out_dir, num_cpus=args.num_cpus)

    all_outputs = {}
    for cid, item in batched_data.items():
        out = predictor.predict({cid: item}, {cid: bs_msa_all[cid]})
        all_outputs.update(out)

    # ===== Print consensus and probabilities =====
    cluster = all_outputs["cluster_0"]
    probs = np.array(cluster["consensus_probs"])  # shape [T, 5]
    base_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "-"}

    # Get max probability and index per position
    max_idx = np.argmax(probs, axis=1)
    max_val = np.max(probs, axis=1)

    # Remove gaps ('-')
    keep_mask = max_idx != 4
    consensus = "".join([base_map[i] for i in max_idx[keep_mask]])
    filtered_probs = max_val[keep_mask].tolist()

    print("\n===== SeqFormer Consensus Result =====")
    print("Consensus:")
    print(consensus)
    print("\nProbabilities:")
    print(filtered_probs)
    print("=======================================\n")


# ---------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="SeqFormer consensus prediction from SEQ+QUAL FASTA")

    parser.add_argument("--input", required=True, help="Input FASTA file with optional quality lines")

    parser.add_argument("--ngs", action="store_true", help="Use Illumina (NGS) model")
    parser.add_argument("--tgs", action="store_true", help="Use Nanopore (TGS) model")

    parser.add_argument("--bsalign-out-dir", default="./bsalign_tmp", help="Temporary directory for BSAlign output")
    parser.add_argument("--context-len", type=int, default=512, help="Context length for model input")
    parser.add_argument("--max-depth", type=int, default=16, help="Maximum number of reads per cluster")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for model inference")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--num-cpus", type=int, default=8, help="Number of CPU threads for BSAlign")

    args = parser.parse_args()
    run_consensus_prediction(args)
