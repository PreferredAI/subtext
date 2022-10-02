import argparse
import csv
import datetime
import json
import logging
import re
import shutil
import string
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from blingfire import text_to_words
from scipy.spatial.distance import cosine
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDINGS, VOCAB, VOCABSET, WPTREE, DECODELOOKUP, EMBLOOKUP = None, None, None, None, {}, {}


def load_sent_data(data_path, window_size=3):
    result = {'tag': [], 'choices': [], 'context': [], 'y': []}
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['tag', 'choices', 'pre', 'post', 'y'], quoting=csv.QUOTE_ALL)
        for example in reader:
            full_pre = example['pre'].translate(str.maketrans('', '', string.punctuation))    # removes punctuation
            full_post = example['post'].translate(str.maketrans('', '', string.punctuation))  # removes punctuation
            context = full_pre.split()[-window_size:] + full_post.split()[:window_size]
            if context:
                result['tag'].append(example['tag'])
                result['choices'].append(example['choices'].split())
                result['y'].append(int(example['y']))
                result['context'].append(context)
    return pd.DataFrame.from_dict(result)


def main(args):
    log_dir = args.log_dir / args.embeddings.stem
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(f'{log_dir}')
    data = json.load(open(args.clinc_data))
    records = {'type': [], 'x': [], 'label': []}
    for k, v in data.items():
        for example in v:
            if example[1] == 'oos':
                continue
            records['type'].append(k.replace('oos_', ''))
            x = text_to_words(example[0])
            records['x'].append(x)
            records['label'].append(example[1])
    label_lookup = {label: idx for idx, label in enumerate(sorted({*records['label']}))}
    logger.info(f'num_labels: {len(label_lookup)}')
    records['y'] = [label_lookup[i] for i in records['label']]
    df = pd.DataFrame.from_dict(records)
    logger.info('records len:' + '\t'.join(f'{type}\t{len(df[df.type == type])}' for type in ['train', 'val', 'test']))

    args.results_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_lookup))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'pytorch device: {device}')
    model.to(device)

    # LOAD SUBTEXT EMBEDDINGS
    _EMBEDDINGS = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                   [line.strip().split(maxsplit=1) for line in islice(open(args.embeddings), 1, None)]}
    EMBEDDINGS = {tok: emb for tok, emb in _EMBEDDINGS.items()}
    VOCAB = {i for i in EMBEDDINGS.keys()}

    if not args.is_vanilla:
        # we only consider k splits if it generates less than args.decode_type[23:] wordpiece permutations
        WPTREE = WordpieceTree.from_vocab(vocab=EMBEDDINGS.keys(),
                                          unknown_piece=args.unk_tok,
                                          decode='all',
                                          decode_k=args.decode_k)
        def _get_tok_emb(tok, context_emb=None):
            min_cosdist, min_cosdist_decomp, min_cosdist_emb = 10, None, None
            all_decompositions = WPTREE.decode(tok)
            if not all_decompositions:
                all_decompositions = [[args.unk_tok]]
            tok_len = len(tok)
            for decomp in all_decompositions:
                decomp_weights = np.array([len(i) for i in decomp]) / tok_len
                decomp_emb = np.array([EMBEDDINGS[wp] if wp in VOCAB else EMBEDDINGS[args.unk_tok] for wp in decomp])
                weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
                cos_dist = cosine(context_emb, weighted_emb)
                if cos_dist < min_cosdist:
                    min_cosdist = cos_dist
                    min_cosdist_decomp = decomp
                    min_cosdist_emb = weighted_emb
            return min_cosdist_emb

        recon_embs = []
        unused_tok_pattern = re.compile(r'^\[unused\d*\]$')
        for tok, original_emb in zip([t for t, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1])],
                                     model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy()):
            if tok == args.eol_tok or unused_tok_pattern.search(tok):
                recon_embs.append(original_emb)
            else:
                recon_embs.append(_get_tok_emb(tok, original_emb))

        with torch.no_grad():
            recon_weights = torch.nn.parameter.Parameter(torch.Tensor(np.stack(recon_embs)).to(device))
            assert model.bert.embeddings.word_embeddings.weight.shape == recon_weights.shape
            model.bert.embeddings.word_embeddings.weight = recon_weights

    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-6)
    num_batched_arrays = int(round(len(df[df.type == 'train']) // args.batch_size, 0))
    num_val_batched_arrays = int(round(len(df[df.type == 'val']) // args.batch_size, 0))
    lr_scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                             num_batched_arrays * 2,
                                                                             num_batched_arrays * args.num_epochs)

    Path(f'{args.results_dir}/best/{args.embeddings.stem}/').mkdir(exist_ok=True, parents=True)
    Path(f'{args.results_dir}/final/{args.embeddings.stem}/').mkdir(exist_ok=True, parents=True)
    best_checkpoint_name, best_checkpoint_val = None, 1E6
    early_stop = False
    tqdm_bar = tqdm(range(1, args.num_epochs+1), total=args.num_epochs)
    for epoch_idx in tqdm_bar:
        train_losses = []
        if early_stop:
            break

        epoch_df = df[df.type == 'train'].sample(frac=1, random_state=epoch_idx)
        epoch_tqdm_bar = enumerate(np.array_split(epoch_df, num_batched_arrays))
        for batch_idx, batch in epoch_tqdm_bar:
            inputs = tokenizer(batch.x.tolist(), return_tensors="pt", padding=True).to(device)
            labels = torch.tensor(batch.y.tolist()).to(device)
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            train_losses.append(loss.item())
            writer.add_scalar('train_loss', loss, (((epoch_idx - 1) * num_batched_arrays) + batch_idx))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_idx % (num_batched_arrays // 4) == 0:
                batch_save_idx = batch_idx // (num_batched_arrays // 4)
                checkpoint_name = f'E{epoch_idx:02d}.{batch_save_idx:01d}'

                with torch.no_grad():
                    val_tqdm_bar = np.array_split(df[df.type == 'val'], num_val_batched_arrays)
                    val_loss = 0
                    for batch in val_tqdm_bar:
                        inputs = tokenizer(batch.x.tolist(), return_tensors="pt", padding=True).to(device)
                        labels = torch.tensor(batch.y.tolist()).to(device)
                        outputs = model(**inputs, labels=labels, return_dict=True)
                        val_loss += outputs.loss.item()
                    writer.add_scalar('val_loss', val_loss, (((epoch_idx - 1) * num_batched_arrays) + batch_idx))
                    logger.info(f'VAL LOSS @ {checkpoint_name}: {val_loss:0.4f} / {len(df[df.type == "val"])} examples')
                    if batch_save_idx == 0 and val_loss < best_checkpoint_val:
                        shutil.rmtree(f'{args.results_dir}/best/{args.embeddings.stem}')  # remove previous checkpoint
                        model.save_pretrained(f'{args.results_dir}/best/{args.embeddings.stem}/{checkpoint_name}')
                        best_checkpoint_name, best_checkpoint_val = checkpoint_name, val_loss
        tqdm_bar.set_postfix_str(f'epoch avg loss: {np.average(train_losses):0.4f}')

    model.save_pretrained(f'{args.results_dir}/final/{args.embeddings.stem}/{checkpoint_name}')

    # TESTING
    del model
    model = BertForSequenceClassification.from_pretrained(str(next(Path(f'{args.results_dir}/best/{args.embeddings.stem}/').glob('E*')).resolve()))
    model.to(device)
    test_losses, test_ranks = [], []
    with torch.no_grad():
        test_df = df[df.type == 'test']
        num_batched_arrays = int(round(len(test_df) // args.batch_size, 0))
        tqdm_bar = tqdm(enumerate(np.array_split(test_df, num_batched_arrays)), total=num_batched_arrays)

        for batch_idx, batch in tqdm_bar:
            inputs = tokenizer(batch.x.tolist(), return_tensors="pt", padding=True).to(device)
            labels = torch.tensor(batch.y.tolist()).to(device)
            outputs = model(**inputs, labels=labels, return_dict=True)
            test_losses.append(outputs.loss)
            for logits, target in zip(outputs.logits, batch.y.tolist()):
                rank = (torch.argsort(logits, descending=True) == target).nonzero().item() + 1
                test_ranks.append(rank)

    rank_dir = args.results_dir / 'output_clinc150_ranks'
    rank_dir.mkdir(exist_ok=True, parents=True)
    np.savetxt(rank_dir / f'{args.embeddings.stem}.npy', np.array(test_ranks))

    logger.info(f'BEST CHECKPOINT LOSS: {best_checkpoint_name}: {best_checkpoint_val:0.4f}')
    logger.info(f'{args.embeddings.stem}\tBEST-CHECKPOINT-LOSS\t{best_checkpoint_name}\t{best_checkpoint_val:0.4f}\t'
                f'eval_loss: {torch.stack(test_losses).sum().item():0.4f}\t'
                f'mrr: {np.mean([1/i for i in test_ranks]):0.4f}\t'
                f'mean perplexity: {torch.exp(torch.stack(test_losses).sum() / len(test_df)).item():0.4f} '
                f'for {len(test_df)} examples')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--eol_tok', type=str, default='</s>')
    argparser.add_argument('--num_epochs', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=250)

    argparser.add_argument('--window_size', type=int, default=3)

    argparser.add_argument('--clinc_data', type=Path, default=Path('../data/raw/clinc150/data_full.json'))
    argparser.add_argument('--embeddings', type=Path, default=Path('../data/bert-base-uncased/bert-base-uncased-25000.wv'))
    argparser.add_argument('--is_vanilla', action='store_true')  # turns off emb reconstruction, lookup layer replacement
    argparser.add_argument('--decode_k', type=int, default=1_000)

    argparser.add_argument('--log_dir', type=Path, default=Path('TBLOGS'))
    argparser.add_argument('--results_dir', type=Path, default=Path('DEBUG'))

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
