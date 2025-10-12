import torch
import torch.cuda
import torch.backends.cudnn
import os
import re
import json
import random
import numpy as np
import collections
import parasail
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scheduler import linear_warmup_cosine_decay
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model_v2 import Model_v2
from data_loader import MAS_Realtime_Dataset_Bsalign_v2


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
    return acc_ * 100


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--sequencing-type', type=str, choices=['illumina', 'ont'], required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--max-norm', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=512, )
    parser.add_argument('--output-dim', type=int, default=5, )
    parser.add_argument('--num-heads', type=int, default=8, )
    parser.add_argument('--max-seq-len', type=int, default=1024, )
    # parser.add_argument('--decode-len', type=int, default=300, )
    parser.add_argument('--num-layers', type=int, default=8, )
    parser.add_argument('--data-limit', type=int, default=50000, )
    parser.add_argument('--min-depth', type=int, default=3, )
    parser.add_argument('--max-depth', type=int, default=None, )
    parser.add_argument('--context-len', type=int, default=400, )  # 训练时动态采样计算 msa, 可能出现更长的比对结果, context_len 尽量取更大的值避免训练失败
    parser.add_argument('--base-vocab', type=int, default=7, )
    # parser.add_argument('--ref-vocab', type=int, default=5, )
    parser.add_argument('--qual-vocab', type=int, default=41, )
    parser.add_argument('--disable-qual', action='store_true', default=False, )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger_path = os.path.join(os.path.abspath(args.output), 'log')
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger = SummaryWriter(logger_path)

    weight_path = os.path.join(os.path.abspath(args.output), 'weight')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    set_global_seed(args.seed)

    # train_dataset = MAS_Realtime_Dataset(
    #     data_path=args.data, total_cnt=args.data_limit, max_depth=args.max_depth, min_depth=args.min_depth, context_len=args.context_len, is_validate=False, train_ratio=0.98,
    # )
    # valid_dataset = MAS_Realtime_Dataset(
    #     data_path=args.data, total_cnt=args.data_limit, max_depth=args.max_depth, min_depth=args.min_depth, context_len=args.context_len, is_validate=True, train_ratio=0.98,
    # )

    train_dataset = MAS_Realtime_Dataset_Bsalign_v2(
        data_path=args.data, total_cnt=args.data_limit, max_depth=args.max_depth, min_depth=args.min_depth, context_len=args.context_len, is_validate=False, train_ratio=0.98, train_record_ratio=0.75, seed=42,
    )
    valid_dataset = MAS_Realtime_Dataset_Bsalign_v2(
        data_path=args.data, total_cnt=args.data_limit, max_depth=args.max_depth, min_depth=args.min_depth, context_len=args.context_len, is_validate=True, train_ratio=0.98, train_record_ratio=0.75, seed=42,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )

    model = Model_v2(
        base_vocab_size=args.base_vocab,
        # ref_vocab_size=args.ref_vocab,
        qual_vocab_size=args.qual_vocab,
        dim=args.dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        # decode_len=args.decode_len,
        num_layers=args.num_layers,
        disable_qual=args.disable_qual,
    )
    model = model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler_fn = linear_warmup_cosine_decay(warmup_steps=500)
    scheduler = scheduler_fn(optimizer, train_loader, args.epochs, 0)

    scaler = torch.GradScaler('cuda', enabled=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        step_count = 0
        # for step, (seqs, quals, refs, lens, record_ids, sample_depths) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        for step, (seqs, quals, refs, lens) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
            global_step = epoch * len(train_loader) + step

            optimizer.zero_grad()
            seqs = seqs.cuda(args.gpu)
            quals = quals.cuda(args.gpu)
            refs = refs.cuda(args.gpu)
            lens = lens.cuda(args.gpu)

            with torch.autocast('cuda', enabled=True):
                loss = model.forward(seqs, quals, refs, lens, return_logits=False)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

            logger.add_scalar('train/loss', loss.item(), global_step + 1)
            logger.add_scalar('train/grad_norm', grad_norm, global_step + 1)
            logger.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step + 1)
            logger.add_scalar('train/scale', scaler.get_scale(), global_step + 1)

        logger.add_scalar('train/avg_loss', total_loss/step_count, epoch + 1)

        model.eval()
        torch.save(model.state_dict(), os.path.join(weight_path, 'epoch_%d.pth' % epoch))

        total_loss = 0
        step_count = 0
        acces = []
        # for step, (seqs, quals, refs, lens, record_ids, sample_depths) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='valid'):
        for step, (seqs, quals, refs, lens) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='valid'):
            seqs = seqs.cuda(args.gpu)
            quals = quals.cuda(args.gpu)
            refs = refs.cuda(args.gpu)
            lens = lens.cuda(args.gpu)

            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True):
                    pred_seqs, _ = model.inference(seqs, quals, lens)

            # total_loss += loss.item()
            # step_count += 1

            for b in range(len(refs)):
                target = refs[b].cpu()
                target = ''.join(['NACGT'[ch] for ch in target])
                pred = pred_seqs[b]
                acc = align_accuracy(ref=target, seq=pred, min_coverage=0.8)
                acces.append(acc)

        # logger.add_scalar('valid/avg_loss', total_loss/step_count, epoch + 1)
        logger.add_scalar('valid/mean_acc', np.mean(acces), epoch + 1)
        logger.add_scalar('valid/median_acc', np.median(acces), epoch + 1)
