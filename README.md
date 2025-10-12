# 🧬 SeqFormer

<p align="center">
  <img src="/seqFormer.png" alt="Polus Logo" width="1600"/>
</p>

**SeqFormer** is a deep learning–based encoder–decoder model for **DNA storage sequence reconstruction** and **error probability estimation**.
 It is designed to recover original sequences from noisy sequencing reads and provide base-level confidence scores across DNA storage channels.

------

## ⚙️ Installation

### 1. Create and activate the environment

```bash
# Create a new conda environment
conda create -n Seqformer python=3.10
conda activate Seqformer

# Install PyTorch and TorchTune (CUDA 11.8 build)
pip install torch==2.6.0 torchao==0.8.0 --index-url https://download.pytorch.org/whl/cu118
pip install torchtune==0.5.0

# Install additional dependencies
pip install -r requirements.txt
```

------

### 2. Build auxiliary reconstruction tools

Enter the `dna_storage_bench_tools/Reconstruction` directory:

```bash
cd dna_storage_bench_tools/Reconstruction
```

#### (1) BSAlign

```bash
cd bsalign
make
cd ..
```

#### (2) BMALA

```bash
cd BMALA
g++ -std=c++11 BMALookahead.cpp -o BMALA
cd ..
```

#### (3) Iterative Reconstruction

```bash
cd Iterative
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o LCS2.o LCS2.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o EditDistance.o EditDistance.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o Clone.o Clone.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o Cluster2.o Cluster2.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o LongestPath.o LongestPath.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o CommonSubstring2.o CommonSubstring2.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o DividerBMA.o DividerBMA.cpp
g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o DNA.o DNA.cpp
g++ -o DNA *.o
```

------

## 📦 Dataset Preparation

SeqFormer supports **direct HDF5 datasets** for training and evaluation.
 You can convert a text-based dataset (e.g., `data25_phred_illumina.txt`) to HDF5 format using:

```
python data_preprocess_v2.py \
  --input ./data/data25_phred_illumina.txt \
  --output ./data/data25_phred_illumina \
  --depth 25
```

This will produce a file:

```
./data/data25_phred_illumina_hdf5
```

------

### 🧬 Example 

**`data25_phred_illumina.txt`**

```
AGTGCAACAAGTCAATCCGTTATATGCATATAGCTGATAGACTGACGAGCACGCATACTGTCTGCTAGATGCTGTACAGCTATCAGATGAGTCTGTGAGTAGCTGTGTCTGTACGTCTCACACGACTATAAATTGAATGCTTGCTTGCCG
****
AGTGCAACAAGTCAATCCGTTATATGCATATAACTAGTGATGACTGTCGTCTACGTGTCTACTAGTGAGTACAGATAGTAGAGCTGCAGCAGCTACGTGCGCGCTAGTATATCATGATCGACGTCGATAGAATTGAATGCTTGCTTGCCT
EEEEAEEEEEEEEEEEEEEEEE<///E/E////<A//EEEE/E//E/</AE<E////EE/EEE//A/AE////////////6/EE/E//<</<////<///E/E//</////<AA///A/</<////6//6A<AAA/EA<AE/6/E/6//
AGTGCAACAAGTCAATCCGTTATATGCATATAACTAGTGATGACTGTCGTCTACGTGTCTACTAGTGAGTACAGATAGTAGAGCTGCAGCAGCTACGTGCGCGCTAGTATATCATGATCGACGTCGATAGAATTGAATGCTTGCTTGCCT
EAEEEAEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
...
```

Each **data block** follows this structure:

1. **Reference sequence** (true target)
2. `****` separator
3. **N pairs** of `(read sequence, quality string)`, where `N = depth`

------

### 📊 Output Structure (HDF5)

The generated `data25_phred_illumina_hdf5` file will contain groups for each block:

```
data25_phred_illumina_hdf5
 ├── 0/
 │    ├── refs    → reference sequence (1)
 │    ├── seqs    → noisy reads (depth)
 │    └── quals   → quality strings (depth)
 ├── 1/
 │    ├── refs
 │    ├── seqs
 │    └── quals
 ├── ...
```

------

### ⚙️ Command-line Arguments

| Argument   | Type | Description                                                 |
| ---------- | ---- | ----------------------------------------------------------- |
| `--input`  | str  | Path to input text file (e.g. `data25_phred_illumina.txt`)  |
| `--output` | str  | Output directory for generated `data25_phred_illumina_hdf5` |
| `--depth`  | int  | Number of reads per block (e.g. 10, 25, 50, etc.)           |

------

## 🚀 Training

To train the SeqFormer model, run:

```bash
LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so \
python train_v2.py \
  --data ./data/data25_phred_illumina \
  --output train_output \
  --batch-size 64 \
  --epochs 10 \
  --lr 0.0002 \
  --gpu 6 \
  --seed 40 \
  --min-depth 5\
  --max-depth 25
```

| Argument         | Default        | Description                                                  |
| ---------------- | -------------- | ------------------------------------------------------------ |
| `--data`         | **(required)** | Path to the dataset directory (e.g., contains `dataset.hdf5`). |
| `--output`       | **(required)** | Directory to save training checkpoints and logs.             |
| `--batch-size`   | `64`           | Batch size used for training. Reduce if GPU memory is limited. |
| `--epochs`       | `10`           | Number of total training epochs.                             |
| `--num-workers`  | `8`            | Number of CPU threads used for DataLoader.                   |
| `--lr`           | `0.0002`       | Learning rate for optimizer.                                 |
| `--gpu`          | `0`            | GPU device ID to use for training (e.g., `--gpu 6`).         |
| `--seed`         | `40`           | Random seed for reproducibility.                             |
| `--max-norm`     | `1.0`          | Gradient clipping maximum norm.                              |
| `--dim`          | `512`          | Hidden dimension size of the Transformer model.              |
| `--output-dim`   | `5`            | Output dimension (corresponds to A, C, G, T, and gap `-`).   |
| `--num-heads`    | `8`            | Number of attention heads in each Transformer layer.         |
| `--max-seq-len`  | `1024`         | Maximum sequence length supported by the model.              |
| `--num-layers`   | `8`            | Number of Transformer encoder–decoder layers.                |
| `--data-limit`   | `50000`        | Maximum number of records to load from the dataset.          |
| `--min-depth`    | `3`            | Minimum number of reads sampled per cluster during training. |
| `--max-depth`    | `None`         | Maximum number of reads sampled per cluster (auto-set if not specified). |
| `--context-len`  | `400`          | Context window size used during training (affects attention span). |
| `--base-vocab`   | `7`            | Vocabulary size for base encoding (A, C, G, T, N, PAD, GAP). |
| `--qual-vocab`   | `41`           | Vocabulary size for Phred-quality encoding (Illumina: 41, Nanopore: 91). |
| `--disable-qual` | *flag*         | If specified, disables quality score input (train only on base sequences). |

------

## 🧪 Evaluation

The trained SeqFormer can be compared against baseline reconstruction methods including:

- **BSAlign**
- **POA**
- **Iterative**
- **BMALA**
- **SeqFormer (ours)**

Evaluation notebooks are provided:

- `model_benchmark-illumina.ipynb`
- `model_benchmark-nanopore.ipynb`



```python
dataset_dict = {
    'illumina': './dataset_illumina.hdf5',
}

model_dict = {
    'illumina': {
        'v2': {
            'all': './model/Illumina.pth',
        }
    }
}
```

### Pretrained Models

| Sequencing Platform | Model Download                                               | Save Path                        |
| ------------------- | ------------------------------------------------------------ | -------------------------------- |
| **Illumina**        | [Google Drive link](https://drive.google.com/file/d/1Hsn9_nFD6RqBiTzLxSv4WgCoalLa7Tfl/view?usp=drive_link) | `./model/seqformer_Illumina.pth` |
| **Nanopore (ONT)**  | [Google Drive link](https://drive.google.com/file/d/1MG6zPakV2Cuvp8NZYjCM4BjiwrFgat9U/view?usp=drive_link) | `./model/seqformer_ONT.pth`      |

Place the downloaded models under the `model/` directory before running the benchmark notebooks.

------



## 🔮 SeqFormer Consensus CLI

### Pretrained Models

| Sequencing Platform | Model Download                                               | Save Path                        |
| ------------------- | ------------------------------------------------------------ | -------------------------------- |
| **Illumina**        | [Google Drive link](https://drive.google.com/file/d/1Hsn9_nFD6RqBiTzLxSv4WgCoalLa7Tfl/view?usp=drive_link) | `./model/seqformer_Illumina.pth` |
| **Nanopore (ONT)**  | [Google Drive link](https://drive.google.com/file/d/1MG6zPakV2Cuvp8NZYjCM4BjiwrFgat9U/view?usp=drive_link) | `./model/seqformer_ONT.pth`      |

Place the downloaded models under the `model/` directory before running the benchmark notebooks.

### Usage

```
python seqformer_predict.py \
  --input ./data/illumina_input.txt \
  --ngs \
  --bsalign-out-dir ./bsalign_tmp \
  --context-len 512 \
  --max-depth 16 \
  --batch-size 4 \
  --gpu 0 \
  --num-cpus 8
```



------

### Input Format

The input FASTA file can contain **reads with or without quality lines**.

#### ✅ Case 1: Reads with quality values (recommended for Illumina/NGS)

Each sequence entry includes both the nucleotide sequence and its ASCII Phred-quality string:

```
>seq1
AGCTGACTGACGT
IIIIIIIIIIIIII
>seq2
AGCTGACTAAGCT
IIIIIIIIIIIIII
```



#### ✅ Case 2: Reads without quality values (typical for Nanopore/TGS)

For long-read datasets (e.g., Oxford Nanopore), quality lines can be omitted:

```
>read1
AGCTGACTGACGT
>read2
AGCTGACTAAGCT
```



```
If no quality line is provided:

- **NGS mode (`--ngs`)** → missing qualities are automatically filled with `"I"` (Phred ≈ 40)  
- **TGS mode (`--tgs`)** → missing qualities are automatically filled with `"Z"` (Phred ≈ 90)

This ensures compatibility with both Illumina and Nanopore datasets without requiring additional preprocessing.
```

------



### Output

The CLI directly prints the **consensus sequence** and **base-wise probabilities** to the terminal:

```
===== SeqFormer Consensus Result =====
Consensus:
AGCTGACTGACGT

Probabilities:
[0.9994, 0.9999, 0.9998, 0.9999, ...]
=======================================
```



## 📘 Citation

If you use SeqFormer in your research, please cite this repository once the associated paper or preprint is released.

------


