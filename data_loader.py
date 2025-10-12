from torch.utils.data import Dataset
import numpy as np
import torch
import os
from spoa import poa
import h5py


def base_converter(ch):
    return {
        'A': 1,
        'C': 2,
        'G': 3,
        'T': 4,
        '-': 5,
        'N': 6,
    }[ch]


def encode_base_seq(seq):
    return np.array(list(map(base_converter, seq.upper())), dtype=np.int64)


def qual_convertion(qual, delta=33):
    return np.frombuffer(qual.encode('ascii'), dtype=np.uint8) - delta


class MAS_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        is_validate=False,
        train_ratio=0.98,
        seed=42,
    ):
        super().__init__()

        self.data = torch.load(os.path.join(data_path, 'dataset.pth.tar'))
        self.seqs = self.data['seqs']
        self.quals = self.data['quals']
        self.lens = self.data['lens']
        self.refs = self.data['refs']

        total_cnt = len(self.lens)
        train_cnt = int(total_cnt * train_ratio)
        valid_cnt = total_cnt - train_cnt

        _indices = np.arange(total_cnt)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(_indices)

        if is_validate:
            if valid_cnt <= 0:
                raise RuntimeError('validation set is empty')
            self.indices = _indices[-valid_cnt:]
            self.dataset_size = valid_cnt
        else:
            if train_cnt <= 0:
                raise RuntimeError('train set is empty')
            self.indices = _indices[:train_cnt]
            self.dataset_size = train_cnt

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            self.seqs[idx].to(torch.long),
            self.quals[idx].to(torch.long),
            self.refs[idx].to(torch.long),
            int(self.lens[idx]),
        )


class MAS_Realtime_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        total_cnt,
        min_depth,
        max_depth=None,
        context_len=378,  # padded reads 的最大长度 (二代/三代测序取值不同) => 在训练和测试时需要保持一致, 以保证位置编码结果正确.
        is_validate=False,
        train_ratio=0.98,
        # seed=42,
    ):
        super().__init__()
        self.data = torch.load(os.path.join(data_path, 'dataset.pth.tar'))
        self.seqs = self.data['seqs']
        self.quals = self.data['quals']
        self.lens = self.data['lens']
        self.refs = self.data['refs']
        self.min_depth = min_depth
        self.max_depth = self.seqs.size(1) if max_depth is None else max_depth
        self.max_length = context_len
        self.is_validate = is_validate
        assert self.min_depth <= self.max_depth

        train_cnt = int(total_cnt * train_ratio)
        valid_cnt = total_cnt - train_cnt

        train_records = int(self.seqs.size(0) * train_ratio)
        valid_records = self.seqs.size(0) - train_records

        if is_validate:
            self.dataset_size = valid_cnt
            self.num_records = valid_records
            self.seqs = self.seqs[-valid_records:]
            self.quals = self.quals[-valid_records:]
            self.lens = self.lens[-valid_records:]
            self.refs = self.refs[-valid_records:]
        else:
            self.dataset_size = train_cnt
            self.num_records = train_records
            self.seqs = self.seqs[:train_records]
            self.quals = self.quals[:train_records]
            self.lens = self.lens[:train_records]
            self.refs = self.refs[:train_records]

        print('number of records:', self.num_records)

    def __len__(self):
        return self.dataset_size

    def extract_data(self, record_idx, seq_idx, length):
        seq: torch.Tensor = self.seqs[record_idx, seq_idx, :length].to(torch.long)
        qual: torch.Tensor = self.quals[record_idx, seq_idx, :length].to(torch.long)
        indices = torch.where(seq != 5)[0]
        seq = seq[indices]
        qual = qual[indices]
        seq_str = ''.join(['0ACGT-N'[ch] for ch in seq])
        return seq_str, qual.numpy()

    def __getitem__(self, i):
        record_idx = np.random.randint(0, self.num_records)
        sampling_num = np.random.randint(self.min_depth, self.max_depth + 1)
        sample_indices = np.random.choice(self.max_depth, size=sampling_num, replace=False)

        curr_ref = self.refs[record_idx].to(torch.long)
        curr_len = int(self.lens[record_idx])
        sample_seqs = []
        sample_quals = []
        for seq_idx in sample_indices:
            sample_seq, sample_qual = self.extract_data(record_idx, seq_idx, curr_len)
            sample_seqs.append(sample_seq)
            sample_quals.append(sample_qual)

        consensus, msa = poa(sample_seqs, algorithm=1)
        new_len = len(msa[0])
        new_seqs = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)
        new_quals = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)
        for i in range(len(msa)):
            ali_seq = encode_base_seq(msa[i])
            ali_qual = np.ones(new_len, dtype=np.int64)
            ali_qual[np.where(ali_seq != 5)[0]] = sample_quals[i]
            new_seqs[i, :new_len] = torch.from_numpy(ali_seq)
            new_quals[i, :new_len] = torch.from_numpy(ali_qual)

        return (
            new_seqs,
            new_quals,
            curr_ref,
            new_len,
            record_idx,
            sampling_num,
        )


def bsalign_process(file_dir, file_idx, seqs):
    bsalign_exe_path = '/home/xsh/bsalign/bsalign'
    fasta_path = os.path.join(file_dir, f"{file_idx}.fasta")
    consensus_path = os.path.join(file_dir, f"{file_idx}.consensus")
    alignment_path = os.path.join(file_dir, f"{file_idx}.alignment")

    with open(fasta_path, 'w') as test_f:
        for s_id, s in enumerate(seqs):
            test_f.write(f'>{s_id}\n{s}\n')

    _ = os.system(
        f'{bsalign_exe_path} poa {fasta_path} -o {consensus_path} -L > {alignment_path}'
    )

    with open(alignment_path) as f:
        msa = [
            line.strip().split(' ')[3].upper().replace('.', '-')
            for line in f.readlines()[2: 2 + len(seqs)]
        ]

    os.remove(fasta_path)
    os.remove(consensus_path)
    os.remove(alignment_path)
    return msa


class MAS_Realtime_Dataset_Bsalign(Dataset):
    """
    使用 BSALIGN_MSA 作为输入, 模型为自回归设计, 训练和推理的多序列比对都不使用 ref
    """
    def __init__(
        self,
        data_path,
        total_cnt,
        min_depth,
        max_depth,
        tmp_dir='./tmp',
        context_len=520,
        is_validate=False,
        train_ratio=0.98,
        train_record_ratio=0.75,
        seed=42,
    ):
        super().__init__()
        self.data = h5py.File(os.path.join(data_path, 'dataset.hdf5'), 'r')
        self.tmp_dir = tmp_dir
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_length = context_len
        self.is_validate = is_validate

        train_cnt = int(total_cnt * train_ratio)
        valid_cnt = total_cnt - train_cnt

        rng = np.random.default_rng(seed=seed)
        total_records = len(self.data.keys())
        total_record_indices = np.arange(total_records)
        rng.shuffle(total_record_indices)
        train_records = int(total_records * train_record_ratio)
        valid_records = total_records - train_records

        if is_validate:
            self.dataset_size = valid_cnt
            self.record_indices = total_record_indices[-valid_records:]
        else:
            self.dataset_size = train_cnt
            self.record_indices = total_record_indices[:train_records]

        print('{} number of records:'.format('Valid' if is_validate else 'Train'), len(self.record_indices))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        record_idx = self.record_indices[np.random.randint(0, len(self.record_indices))]
        sampling_num = np.random.randint(self.min_depth, self.max_depth + 1)
        sample_indices = np.sort(np.random.choice(self.max_depth, size=sampling_num, replace=False))

        curr_ref = encode_base_seq(self.data[str(record_idx)]['refs'][0].decode("ascii"))
        sample_seqs = [s.decode('ascii') for s in self.data[str(record_idx)]['seqs'][sample_indices]]
        sample_quals = [s.decode('ascii') for s in self.data[str(record_idx)]['quals'][sample_indices]]

        msa = bsalign_process(file_dir=self.tmp_dir, file_idx=i, seqs=sample_seqs)
        new_len = len(msa[0])
        new_seqs = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)
        new_quals = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)

        for i in range(len(msa)):
            ali_seq = encode_base_seq(msa[i])
            ali_qual = np.ones(new_len, dtype=np.int64)
            ali_qual[np.where(ali_seq != 5)[0]] = qual_convertion(sample_quals[i]).astype(np.int64)
            new_seqs[i, :new_len] = torch.from_numpy(ali_seq)
            new_quals[i, :new_len] = torch.from_numpy(ali_qual)

        return (
            new_seqs,
            new_quals,
            curr_ref,
            new_len,
        )


class MAS_Realtime_Dataset_Bsalign_v2(Dataset):
    """
    使用 BSALIGN_MSA 作为输入, 模型为非自回归设计, 训练时多序列比对额外增加 ref 的信息.
    """
    def __init__(
        self,
        data_path,
        total_cnt,
        min_depth,
        max_depth,
        tmp_dir='./tmp',
        context_len=520,
        is_validate=False,
        train_ratio=0.98,
        train_record_ratio=0.75,
        seed=42,
    ):
        super().__init__()
        self.data = h5py.File(os.path.join(data_path, 'dataset.hdf5'), 'r')
        self.tmp_dir = tmp_dir
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_length = context_len
        self.is_validate = is_validate

        train_cnt = int(total_cnt * train_ratio)
        valid_cnt = total_cnt - train_cnt

        rng = np.random.default_rng(seed=seed)
        total_records = len(self.data.keys())
        total_record_indices = np.arange(total_records)
        rng.shuffle(total_record_indices)
        train_records = int(total_records * train_record_ratio)
        valid_records = total_records - train_records

        if is_validate:
            self.dataset_size = valid_cnt
            self.record_indices = total_record_indices[-valid_records:]
        else:
            self.dataset_size = train_cnt
            self.record_indices = total_record_indices[:train_records]

        print('{} number of records:'.format('Valid' if is_validate else 'Train'), len(self.record_indices))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        record_idx = self.record_indices[np.random.randint(0, len(self.record_indices))]
        sampling_num = np.random.randint(self.min_depth, self.max_depth + 1)
        sample_indices = np.sort(np.random.choice(self.max_depth, size=sampling_num, replace=False))

        ref_seq = self.data[str(record_idx)]['refs'][0].decode("ascii")
        sample_seqs = [s.decode('ascii') for s in self.data[str(record_idx)]['seqs'][sample_indices]]
        sample_quals = [s.decode('ascii') for s in self.data[str(record_idx)]['quals'][sample_indices]]

        if self.is_validate:
            msa = bsalign_process(file_dir=self.tmp_dir, file_idx=i, seqs=sample_seqs)
        else:
            msa = bsalign_process(file_dir=self.tmp_dir, file_idx=i, seqs=(sample_seqs + [ref_seq]))

        new_len = len(msa[0])
        new_seqs = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)
        new_quals = torch.zeros((self.max_depth, self.max_length), dtype=torch.long)
        for i in range(len(sample_indices)):
            ali_seq = encode_base_seq(msa[i])
            ali_qual = np.ones(new_len, dtype=np.int64)
            ali_qual[np.where(ali_seq != 5)[0]] = qual_convertion(sample_quals[i]).astype(np.int64)
            new_seqs[i, :new_len] = torch.from_numpy(ali_seq)
            new_quals[i, :new_len] = torch.from_numpy(ali_qual)

        if self.is_validate:
            curr_ref = encode_base_seq(ref_seq)
        else:
            curr_ref = torch.zeros(self.max_length, dtype=torch.long)
            curr_ref[:new_len] = torch.from_numpy(encode_base_seq(msa[-1]))

        return (
            new_seqs,
            new_quals,
            curr_ref,
            new_len,
        )
