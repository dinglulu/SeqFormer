import os
import math
import re
import mappy
import collections
import parasail
import numpy as np
import pysam
import shutil
from Bio import SeqIO
from tqdm import tqdm
from glob import glob
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from p_tqdm import p_imap, p_uimap


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


def align_accuracy(ref, seq, balanced=False, min_coverage=0.0):
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
    return acc_ * 100


def align_accuracy_2(alignment, seq, return_metas=False):
    counts = collections.defaultdict(int)
    _, cigar = parasail_to_sam(alignment, seq)

    for count, op in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    acc_ = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])

    if return_metas:
        return acc_ * 100, counts['='], counts['I'], counts['X'], counts['D']

    return acc_ * 100


file_adaptors = {
    'F': [
        'CCGGCACATGAATTCCACTG',
        'CCGACGATGGAGTTTATGCT',
        'GTCTTGCCGATACCGTATGC',
        'GTTTCTGGCCTGCATACACC',
        'GCAATCGCGACACTTAGGTT'
    ],
    'R': [
        'CGAAGATGGCGGTACACCTA',
        'CGCGCACTCTACGTTGATAC',
        'TACGCCGCACATCATACAAC',
        'TTAAAGCAACCGTGGCCTTC',
        'CCTCGTGTCATCTCGGATGA'
    ],
}


file_names = [
    'Birds',
    'Olympics',
    'Szu',
    'BookOfSongs',
    'Voyager',
]


pair_G_primers = {
    'G1F': ('AACGCACCATACGCGCTGATTATGATACTG', mappy.revcomp('GTTGATCGAGTGCCTTCAGTGGTGCAGATA')),
    'G1R': ('GTTGATCGAGTGCCTTCAGTGGTGCAGATA', mappy.revcomp('AACGCACCATACGCGCTGATTATGATACTG')),
    'G2F': ('TATCTGCACCACTGAAGGCACTCGATCAAC', mappy.revcomp('CTAGAGCAGCATCAACATCATAAGACGCCT')),
    'G2R': ('CTAGAGCAGCATCAACATCATAAGACGCCT', mappy.revcomp('TATCTGCACCACTGAAGGCACTCGATCAAC')),
    'G3F': ('AGGCGTCTTATGATGTTGATGCTGCTCTAG', mappy.revcomp('ACCGTCATCACAGACAAGACTTGGTAAGGC')),
    'G3R': ('ACCGTCATCACAGACAAGACTTGGTAAGGC', mappy.revcomp('AGGCGTCTTATGATGTTGATGCTGCTCTAG')),
    'G4F': ('GCCTTACCAAGTCTTGTCTGTGATGACGGT', mappy.revcomp('CCTTCTGTGACTTCACTGCACGGCGACCAT')),
    'G4R': ('CCTTCTGTGACTTCACTGCACGGCGACCAT', mappy.revcomp('GCCTTACCAAGTCTTGTCTGTGATGACGGT')),
    'G5F': ('ATGGTCGCCGTGCAGTGAAGTCACAGAAGG', mappy.revcomp('GTATGTGAACGTACAAGCAGAACTCGTAGT')),
    'G5R': ('GTATGTGAACGTACAAGCAGAACTCGTAGT', mappy.revcomp('ATGGTCGCCGTGCAGTGAAGTCACAGAAGG')),
    'G6F': ('ACTACGAGTTCTGCTTGTACGTTCACATAC', mappy.revcomp('TGAACGTACAAGCAGGCTCGACGCATGTAC')),
    'G6R': ('TGAACGTACAAGCAGGCTCGACGCATGTAC', mappy.revcomp('ACTACGAGTTCTGCTTGTACGTTCACATAC')),
}


def segment_dna_sequence(item, min_G_primer_acc=90, accept_acc=80, min_segment_length=270, max_segment_length=330, ):
    rid, read, qual = item
    _ref_seq = read

    segment_results = []
    for G_primer_id, (G_primer_start, G_primer_end) in pair_G_primers.items():
        alignment_start = parasail.sw_trace_striped_32(G_primer_start, _ref_seq, 8, 4, parasail.dnafull)
        primer_start_end_ref = alignment_start.end_ref
        primer_start_acc = align_accuracy_2(alignment_start, G_primer_start)
        if primer_start_acc < min_G_primer_acc:
            continue

        alignment_end = parasail.sw_trace_striped_32(G_primer_end, _ref_seq, 8, 4, parasail.dnafull)
        primer_end_end_ref = alignment_end.end_ref
        primer_end_acc, matches, insertes, mismatches, deletes = align_accuracy_2(alignment_end, G_primer_end, return_metas=True)
        if primer_end_acc < min_G_primer_acc:
            continue

        if primer_end_end_ref <= primer_start_end_ref:
            continue

        segment_start = primer_start_end_ref + 1
        segment_end = (primer_end_end_ref + 1) - (matches + mismatches + deletes)
        segment_len = segment_end - segment_start

        if segment_len < min_segment_length or segment_len > max_segment_length:
            continue

        top_seg_bases = _ref_seq[segment_start: segment_end][:20]
        search_adaptors = None
        if 'F' in G_primer_id:
            # 正向胶水引物
            search_adaptors = file_adaptors['F']
        else:
            # 反向互补
            search_adaptors = file_adaptors['R']

        max_acc = 0
        max_acc_file_idx = 0
        for file_idx, _adaptor in enumerate(search_adaptors):
            acc_ = align_accuracy(_adaptor, top_seg_bases, min_coverage=0.8)
            if acc_ > max_acc:
                max_acc = acc_
                max_acc_file_idx = file_idx

        if max_acc < accept_acc:
            continue

        segment_results.append((
            segment_start, segment_end, max_acc_file_idx, bool('F' in G_primer_id)
        ))

    if len(segment_results) == 0:
        return None
    else:
        return rid, read, qual, segment_results


def segment_process(bam_f, output_dir, min_quality=10, thread_number=64):
    db = []
    total_reads = 0
    low_qual_reads = 0
    with pysam.AlignmentFile(bam_f, check_sq=False) as bam:
        for read in tqdm(bam, desc='read bam file'):
            total_reads += 1
            read_id = read.query_name
            seq = read.query_sequence
            qual_arr = np.array(read.query_qualities)
            qual_str = bytes(qual_arr + 33).decode('ascii')
            mean_qs = np.mean(qual_arr.astype(np.float32))
            if mean_qs < min_quality:
                low_qual_reads += 1
                continue
            db.append((
                read_id, seq, qual_str
            ))

    print('='*80)
    print('Total reads: %d' % total_reads)
    print('Low quality reads: %d' % low_qual_reads)
    print('DB reads: %d' % len(db))

    file_handles = []
    for fname in file_names:
        file_handles.append(open(os.path.join(output_dir, f"{fname}.fastq"), 'w'))

    iterator = p_uimap(
        segment_dna_sequence,
        db,
        num_cpus=thread_number,
    )

    for results in iterator:
        if results is None:
            continue

        rid, read, qual, seg_results = results
        for seg_st, seg_en, file_idx, is_forward_strand in seg_results:
            seg_read = read[seg_st:seg_en]
            seg_qual = qual[seg_st:seg_en]
            seg_strand = "+"
            if not is_forward_strand:
                seg_read = mappy.revcomp(seg_read)
                seg_qual = seg_qual[::-1]
                seg_strand = "-"

            file_handles[file_idx].write(f'@{rid} seg_st:seg_en:seg_len={seg_st}:{seg_en}:{seg_en - seg_st} ori_strand={seg_strand}\n')
            file_handles[file_idx].write(f'{seg_read}\n')
            file_handles[file_idx].write('+\n')
            file_handles[file_idx].write(f'{seg_qual}\n')
            file_handles[file_idx].flush()

    for handle in file_handles:
        handle.close()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pod5-root', type=str, required=True, help='Root directory of POD5 sequencing reads')
    parser.add_argument('--out-root', type=str, required=True, help='Root directory of output files')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch file number for basecalling & segment')
    parser.add_argument('--gpus', type=str, default="4", help='GPU device or device list separated by comma')
    parser.add_argument('--model', type=str, default='hac', choices=['fast', 'hac', 'sup'], help='Basecalling model (fast or hac or sup)')
    parser.add_argument('--min-quality', type=int, default=10, help='Minimum mean quality of reads')
    parser.add_argument('--thread-number', type=int, default=64, help='Number of threads for segmentation')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches for segmentation')

    args = parser.parse_args()
    pod5_root = args.pod5_root
    out_root = args.out_root
    max_batches = args.max_batches

    if not os.path.exists(pod5_root):
        raise FileNotFoundError(f'POD5 directory not found: {pod5_root}')

    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    pod5_files = [Path(x) for x in glob(pod5_root + "/" + "**/*.pod5", recursive=True)]
    num_batches = math.ceil(len(pod5_files) / args.batch_size)
    print('Total number of pod5 files: {}'.format(len(pod5_files)))
    print('Total number of batches: {}'.format(num_batches))

    for batch_idx in tqdm(range(num_batches)):
        pod5_dir = os.path.join(out_root, f"{batch_idx}_pod5s")
        os.makedirs(pod5_dir, exist_ok=True)
        for pod5_file_path in pod5_files[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, len(pod5_files))]:
            print(pod5_file_path)
            command = f"cp {pod5_file_path} {pod5_dir}"
            _ = os.system(command)

        bam_dir = os.path.join(out_root, f"{batch_idx}_bam")
        os.makedirs(bam_dir, exist_ok=True)
        calling_command = f"dorado basecaller --recursive --device cuda:{args.gpus} --no-trim --models-directory /home/xsh/dorado/models {args.model} {pod5_dir} > {bam_dir}/output.bam"
        _ = os.system(calling_command)

        seg_dir = os.path.join(out_root, f"{batch_idx}_seg")
        os.makedirs(seg_dir, exist_ok=True)
        segment_process(
            bam_f = os.path.join(bam_dir, "output.bam"),
            output_dir = seg_dir,
            min_quality = args.min_quality,
            thread_number = args.thread_number,
        )

        with open(os.path.join(out_root, f"{batch_idx}.finish"), 'w') as f:
            pass

        shutil.rmtree(pod5_dir)
        shutil.rmtree(bam_dir)
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
