"""
1) 三代数据需要先切割处理; => 用参考序列来切 (理论上应该用文件Index)
2) 训练优先平均质量分数高的序列; => 训练用固定的测序深度(>=5, 具体看数据集的长度分布情况); 测试测序深度不固定 (可能 3~10)
3) 长度不一定要限制, 二代可能有偏短的序列; 三代可能有偏长的序列(需要切割);
   对于二代/三代, 可能在训练和测试时都优先进行长度筛选, 因为这个在实际DNA存储解码时应该也是需要考虑的, 具体是否用[长度筛选]要看进行筛选后, 会丢失多少聚类
4) 注意文件的碱基可能有简并碱基和大小写, 对于非 A/T/C/G 的碱基一律视为 gap 碱基 => 在多序列比对前可以先保留下来, 先不干扰多序列比对, 多序列比对后再用 gap 碱基 "-" 替换 N 碱基, 对应位置的质量分数不变.
5) 碱基和质量分数的 padding 符号均为 0;
   gap 碱基的质量分数可以设置为 1("字符) 或者 2 (#字符), 一般比较常用的是用 # 表示最差的情况.
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


split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr


def get_alignment_region(ref, seq, verbose=False):
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    cigstr = alignment.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    rstart = alignment.cigar.beg_ref
    rend = alignment.end_ref + 1  # exclude
    qstart = alignment.cigar.beg_query
    qend = alignment.end_query + 1  # exclude

    if first_op == 'I':
        qstart += int(first_count)
    elif first_op == 'D':
        rstart = int(first_count)

    if verbose:
        print(os.linesep.join([
            alignment.traceback.ref, alignment.traceback.comp, alignment.traceback.query,
        ]))

    return (rstart, rend, rend - rstart), (qstart, qend, qend - qstart)


def align_accuracy(ref, seq, balanced=False, min_coverage=0.5):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = collections.defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        acc_ = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        acc_ = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return acc_ * 100, alignment, cigar


def qual_convertion(qual, delta=33):
    return np.frombuffer(qual.encode('ascii'), dtype=np.uint8) - delta


DNA_SEGMENT_LENGTH = 300

SEQUENCING_ERROR_RATE = {
    'illumina': 0.004,
    'ont': 0.1,
}


def base_converter(ch):
    # padding 字符用 0; gap 字符用 5; 简并碱基用 6
    # 在这个字符系统下,
    # 1) 输入的碱基序列字符表大小为7, 即: 0(padding), 1,2,3,4 (A/C/G/T), 5 (-), 6(N)
    # 2) 输出的参考序列字符表有几种情况 (注意简并碱基不可能出现在参考序列中)
    #   a. 非自回归且参考序列与测序序列一起进行多序列比对: A/C/G/T/-, 用 0/1/2/3/4 进行编码, padding 不参与loss计算;
    #   b. 自回归的模式, 参考序列是独立字符串, 此时有效字符只有 A/C/G/T (1/2/3/4 表示), SOS 用 0 表示, EOS 用 5 表示, 后续 padding 字符也是 [EOS], 即: SOS, ATCGTT..., EOS, EOS, ..
    #   c. 非自回归 + MaskedDiffusion, 和 b. 类似, 但不需要 SOS 字符.
    return {
        'A': 1,
        'C': 2,
        'G': 3,
        'T': 4,
        '-': 5,
        'N': 6,
    }[ch]


def encode_base_seq(seq):
    return np.array(list(map(base_converter, seq.upper())), dtype=np.uint8)


def pad_array(arr, new_l, padding_idx=0):
    new_arr = []
    for item in arr:
        new_item = np.full(new_l, fill_value=padding_idx, dtype=item.dtype)
        new_item[:len(item)] = item
        new_arr.append(new_item)
    return np.array(new_arr)


def is_gzipped(filename):
    with open(filename, 'rb') as fb:
        return fb.read(2) == b'\x1f\x8b'  # Gzip magic number


def smart_open(file_path, mode='rt', encoding='utf-8'):
    if is_gzipped(file_path):
        return gzip.open(file_path, mode, encoding=encoding)
    else:
        return open(file_path)


if __name__ == '__main__':
    """
    TODO /data05/xsh_data/deep_consensus_dataset/nanopore/contigs_ge5.txt.gz (22949块), 但其中最后一块的reads数据误丢失,
    平均一块大概有 4000 条序列, 为了控制数据量, 先设置 record 的数量固定为 20000, 然后每个块设置为 -D 100, 选择质量最高的 100 条序列,
    实际训练时, 从 100 中抽样 30 ~ 100 条任意序列构建当前训练batch. (对于二代数据, 可以考虑抽样 5 ~ 10) 
    
    Command: python data_preprocess.py -i /data05/xsh_data/deep_consensus_dataset/nanopore/contigs_ge5.txt.gz -o /data05/xsh_data/deep_consensus_dataset/nanopore/contigs_ge5_D100_data -m 20000 -s ont -D 100
    """

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--max-records', type=int, default=None)
    parser.add_argument('-s', '--sequencing', type=str, choices=['illumina', 'ont'], required=True)
    parser.add_argument('-D', '--depth', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise RuntimeError('Input file does not exist: {}'.format(args.input))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    min_require_len = DNA_SEGMENT_LENGTH - int(SEQUENCING_ERROR_RATE[args.sequencing] * DNA_SEGMENT_LENGTH)
    max_require_len = DNA_SEGMENT_LENGTH + int(SEQUENCING_ERROR_RATE[args.sequencing] * DNA_SEGMENT_LENGTH)
    depth = args.depth

    print('Configuration:')
    print('min_require_len:', min_require_len)
    print('max_require_len:', max_require_len)
    print('depth:', depth)

    seqs, quals, refs, lens, ranges = [], [], [], [], []
    records = 0
    line_type = 'ref'
    skip_qual = False
    with smart_open(args.input) as f, tqdm(total=args.max_records) as pbar:
        for line in f:
            line = line.strip()
            if line_type == 'ref':
                refs.append(line)
                line_type = '*'
            elif line_type == '*':
                line_type = 'seq'
                continue
            elif len(line) == 0:
                records += 1
                pbar.update(1)
                if args.max_records is not None and records >= args.max_records:
                    break
                line_type = 'ref'
            elif line_type == 'seq':
                if len(seqs) < len(refs):
                    seqs.append([])
                    lens.append([])
                    if args.sequencing == 'ont':
                        ranges.append([])
                if args.sequencing == 'illumina':
                    if min_require_len <= len(line) <= max_require_len:
                        seqs[records].append(line)
                        lens[records].append(len(line))
                        skip_qual = False
                    else:
                        skip_qual = True
                else:
                    ref_ranges, seq_ranges = get_alignment_region(refs[records], line)
                    if min_require_len <= seq_ranges[2] <= max_require_len:
                        ranges[records].append(seq_ranges)
                        seqs[records].append(line[seq_ranges[0]: seq_ranges[1]])
                        lens[records].append(seq_ranges[2])
                        skip_qual = False
                    else:
                        skip_qual = True
                line_type = 'qual'
            elif line_type == 'qual':
                if len(quals) < len(refs):
                    quals.append([])
                if not skip_qual:
                    if args.sequencing == 'illumina':
                        quals[records].append(qual_convertion(line))
                    else:
                        seq_ranges = ranges[records][-1]
                        quals[records].append(qual_convertion(line)[seq_ranges[0]: seq_ranges[1]])
                line_type = 'seq'

    print('total records:', len(refs))
    print('total reads:', np.sum([len(item) for item in seqs]))
    print(f'num records with n_reads < {depth}:', np.sum([1 if len(item) < depth else 0 for item in seqs]))

    ali_refs = []
    ali_ref_lens = []
    ali_seqs = []
    ali_quals = []
    ali_lens = []
    for idx in tqdm(range(len(refs))):
        if len(quals[idx]) < depth:
            continue

        ali_refs.append(encode_base_seq(refs[idx]))
        ali_ref_lens.append(len(refs[idx]))

        mean_qs = np.array([np.mean(qual.astype(np.float32)) for qual in quals[idx]])
        topk_indices = np.argsort(mean_qs)[-depth:][::-1]
        topk_seqs = [seqs[idx][i] for i in topk_indices]
        topk_quals = [quals[idx][i] for i in topk_indices]
        consensus, msa = poa(topk_seqs, algorithm=1)

        for msa_seq, qs_arr in zip(msa, topk_quals):
            ali_seq = encode_base_seq(msa_seq)
            ali_seqs.append(ali_seq)

            ali_qual = np.ones(len(ali_seq), dtype=np.uint8)  # gap base use Q=1
            ali_qual[np.where(ali_seq != 5)[0]] = qs_arr
            ali_quals.append(ali_qual)
        ali_lens.append(len(msa[0]))  # 同一组内的序列只需要保留一个长度值

    ali_ref_lens = np.array(ali_ref_lens)
    ali_lens = np.array(ali_lens)

    max_ref_l = np.max(ali_ref_lens)
    max_seq_l = np.max(ali_lens)
    ali_refs = pad_array(ali_refs, max_ref_l, padding_idx=0)
    ali_seqs = pad_array(ali_seqs, max_seq_l, padding_idx=0)
    ali_quals = pad_array(ali_quals, max_seq_l, padding_idx=0)

    torch.save({
        'refs': torch.from_numpy(ali_refs),
        'seqs': torch.from_numpy(ali_seqs).view(-1, depth, max_seq_l),
        'quals': torch.from_numpy(ali_quals).view(-1, depth, max_seq_l),
        'lens': torch.from_numpy(ali_lens)
    }, os.path.join(args.output, 'dataset.pth.tar'))
