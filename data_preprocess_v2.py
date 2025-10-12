"""
Direct read data_xx_phred.txt to HDF5 dataset without any msa preprocessing => MSA should be done in training
"""
import os
import gzip
import re
import torch
import torch.nn as nn
import collections
import numpy as np
import parasail
import pandas as pd
import collections
from tqdm import tqdm
from spoa import poa
from argparse import ArgumentParser
import h5py


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--depth', type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    out_path = os.path.join(args.output, 'dataset.hdf5')
    out_f = h5py.File(out_path, 'w')

    lines = [line.strip() for line in open(args.input, 'r').readlines()]

    blocks = []
    _cur_block = []
    for i, line in enumerate(lines):
        if len(line) == 0:
            if len(_cur_block) > 0:
                blocks.append(_cur_block)
            _cur_block = []
        else:
            _cur_block.append(line)

    print("blocks:", len(blocks))
    clean_blocks = []
    fail_blocks = 0
    invalid_blocks = 0
    len_invalid_blocks = 0
    for block in blocks:
        if len(block) != (args.depth * 2 + 2):
            fail_blocks += 1
        else:
            invalid_flag = False
            len_invalid_flag = False

            for _idx in range(2, len(block), 2):
                if set(block[_idx]) > set("ACGT"):
                    invalid_flag = True
                    break
                if len(block[_idx]) != len(block[_idx + 1]):
                    len_invalid_flag = True
                    break

            if invalid_flag:
                invalid_blocks += 1
            elif len_invalid_flag:
                len_invalid_blocks += 1
            else:
                clean_blocks.append(block)

    print("fail blocks:", fail_blocks)
    print("invalid blocks:", invalid_blocks)
    print("len invalid blocks:", len_invalid_blocks)
    print('cleaning blocks:', len(clean_blocks))

    # single_chunk_no = (args.depth * 2 + 4)
    # assert len(lines) % single_chunk_no == 0
    # num_chunks = len(lines) // single_chunk_no

    for chunk_idx, chunks_data in tqdm(enumerate(clean_blocks)):
        # chunks_data = lines[chunk_idx * single_chunk_no: (chunk_idx + 1) * single_chunk_no]
        ref = chunks_data[0]
        if chunks_data[1] != "****":
            raise RuntimeError('Format error')
        seqs = []
        quals = []
        for i in range(args.depth):
            seqs.append(chunks_data[2 + 2*i])
            quals.append(chunks_data[2 + 2*i + 1])
        grp = out_f.create_group(str(chunk_idx))
        grp.create_dataset('refs', data=[ref])
        grp.create_dataset('seqs', data=seqs)
        grp.create_dataset('quals', data=quals)

    out_f.close()
