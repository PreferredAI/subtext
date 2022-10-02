<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/Logo(S).png">
  <source media="(prefers-color-scheme: light)" srcset="./img/Logo Inverse(S).png">
  <img alt="Preferred.AI" src="./img/Logo(S).png">
</picture>

# Morphologically-Aware Vocabulary Reduction of Word Embeddings

This repository contains the code for training SubText, as well as scripts for evaluating the performance on various tasks.

We also provide a pre-trained version of SubText, as well as the Word2Vec (Mikolov et al., 2013) embeddings used for training.

Via [Git LFS](https://git-lfs.github.com)
```shell
git lfs pull -I "pretrained/pretrained.tar.gz"
cd pretrained/
tar -xzvf pretrained.tar.gz
```

Via [Dropbox](https://www.dropbox.com)
```shell
cd pretrained/
wget https://www.dropbox.com/s/ha45ck0hjefdbme/pretrained.tar.gz
tar -xzvf pretrained.tar.gz
```

## Usage

We recommend creating a conda environment named `subtext` with the provided [`environment.yml`](environment.yml):
```shell
conda env create -f environment.yml
```

All scripts should be run from the [`src`](/src) directory:
```shell
cd src/
```

---
### Training Subtext

The following commands trains SubText (to 1000 wordpieces) on embeddings in a Word2Vec-format file (`REPLACE_WITH_WV`), and saves them in the given directory (`REPLACE_WITH_DIR`):

```shell
python subtext/train_subtext.py --embeddings REPLACE_WITH_WV --out_dir REPLACE_WITH_DIR --wp_size 1000 
```

SubText of arbitrary sizes can then be generated from the training record file (`*.rec`) with the following commands:

```shell
python subtext/recon_records.py --records REPLACE_WITH_RECORD 
```

### Experiments

The evaluation code can be found in the `experiments` directory, and can be run in a similar fashion:
```shell
python experiments/1_eval_word_reconstruction.py --piece_embs REPLACE_WITH_WV
```

Be sure to update the arguments with the appropriate values (use `-h` to check arguments for each experiment):
```shell
python experiments/1_eval_word_reconstruction.py -h
```

To run other experiments, replace `1_eval_word_reconstruction.py` with the desired experiment.

## Resources

Scripts for downloading and processing evaluation datasets can be found in [`data/`](data).

### Word2Vec (English)
[Word2Vec](https://code.google.com/archive/p/word2vec/) was trained on English Wikipedia, using the [provided CirrusSearch dumps](https://dumps.wikimedia.org/other/).
The pretrained Word2Vec embeddings were trained on `enwiki-20211011-cirrussearch-content_masked.json`.

### Polyglot (Multilingual Word2Vec)
Pretrained language-specific embeddings are available from the [Polyglot project page](https://sites.google.com/site/rmyeid/projects/polyglot).

### 20News
We use the [`20news-bydate.tar.gz`](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) dataset from the [20Newsgroups project page](http://qwone.com/~jason/20Newsgroups/)

### arXiv
We use the [`arxivData.json`](https://github.com/NeelShah18/arxivData/blob/master/arxivData.json) dataset. Details about the dataset can be found on the [Kaggle dataset page](https://www.kaggle.com/neelshah18/arxivdataset).

### BioASQ
We use the [Task A Training (v.2019)](http://participants-area.bioasq.org/datasets/) dataset, which is only accessible after registering as a participant on the [BioASQ Challenge website](http://participants-area.bioasq.org/accounts/login).

### CLINC150
We use the [`data_full.json`](https://github.com/clinc/oos-eval/tree/master/data) dataset from the [project GitHub repository](https://github.com/clinc/oos-eval).


## Citiation
Our paper can be cited in the following formats:

### APA
```text
Chia, C., Tkachenko, M., & Lauw, H. (2022). Morphologically-Aware Vocabulary Reduction of Word Embeddings. In 21st IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology.
```

### Bibtex
```bibtex
@inproceedings{chia2022morphological,
    title={Morphologically-Aware Vocabulary Reduction of Word Embeddings},
    author={Chia, Chong Cher and Tkachenko, Maksim and Lauw, Hady W},
    booktitle={21st IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology},
    year={2022},
    organization={IEEE}
}
```
