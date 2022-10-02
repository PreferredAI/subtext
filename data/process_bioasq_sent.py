import argparse
import csv
import datetime
import json
import logging
import random
import string
import subprocess
from collections import Counter, defaultdict, namedtuple
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

from blingfire import text_to_sentences, text_to_words
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger('process_arxiv_sent')
logger.setLevel(logging.DEBUG)


def output_examples(file_prefix, all_examples, num_samples=5, dedupe=True):
    if dedupe:
        def _group_examples_by_tok(_examples):
            examples_by_token = defaultdict(list)
            [examples_by_token[e.tok].append(e) for e in _examples]
            return dict(examples_by_token)

        all_examples_by_tok = {tag: _group_examples_by_tok(examples) for tag, examples in all_examples.items()}

        tag_counts = Counter([example[0].tags for examples in all_examples_by_tok.values() for example in examples.values()])
        min_count = min(tag_counts.values())
        if min_count < 100:
            target_class_size = 10 * round(min_count / 20)  # half of min, rounded up to nearest 10
        else:
            target_class_size = 100 * round(min_count / 200)  # half of min, rounded up to nearest 100
        logger.info(f'{Path(file_prefix).stem}\t{target_class_size} <= {min_count}\t{" ".join([f"{k}_{v}" for k, v in tag_counts.items()])}')
    else:
        tag_counts = Counter([example.tag for examples in all_examples.values() for example in examples])
        target_class_size = min(tag_counts.values())
    num_tags = len(tag_counts)
    if target_class_size > 300:
        logger.warning(f'current target class size: {target_class_size} ({num_tags * target_class_size}), trunc to 300 ({num_tags * 300})')
        target_class_size = 300

    def _output_examples_to_file(out_filename, output_examples):
        with open(out_filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['tag', 'choices', 'pre', 'post', 'y'], quoting=csv.QUOTE_ALL, extrasaction='ignore')
            for tag in tqdm(output_examples.keys(), total=len(output_examples)):
                writer.writerows(output_examples[tag])

    for sample_idx in range(num_samples):
        sample_rand = random.Random(42 + sample_idx)
        if dedupe:
            balanced_tag_examples = {}
            for tag, tag_examples_by_tok in all_examples_by_tok.items():
                deduped_examples = [sample_rand.sample(tok_examples, 1)[0] for tok_examples in tag_examples_by_tok.values()]
                balanced_tag_examples[tag] = sample_rand.sample(deduped_examples, target_class_size)
        else:
            balanced_tag_examples = {tag: sample_rand.sample(all_examples[tag], target_class_size)
                                     for tag in all_examples.keys()}

        counter_toks = {tag: [e.tok for e in examples] for tag, examples in balanced_tag_examples.items()}

        def _get_choices(tag, example):
            choices = [sample_rand.choice(tag_counter_tok) if key != tag else example.tok
                       for key, tag_counter_tok in counter_toks.items()]
            shuffled_choices = sample_rand.sample(choices, k=num_tags)
            return {'tag': tag,
                    'choices': ' '.join(shuffled_choices),
                    'pre': ' '.join(example.pre),
                    'post': ' '.join(example.post),
                    'y': shuffled_choices.index(example.tok)}

        balanced_out_examples = {tag: [_get_choices(tag, e) for e in examples]
                                 for tag, examples in balanced_tag_examples.items()}

        result_filename = f'{file_prefix}_balanced_sample_{sample_idx:02d}.all'
        _output_examples_to_file(result_filename, balanced_out_examples)


def process_bioasq(doc):
    abstactText = doc['abstractText']
    summary = text_to_sentences(abstactText)
    results = []
    for sent in summary.split('\n'):
        sent = sent.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        sent_toks = text_to_words(sent).lower().split()
        for i in range(len(sent_toks)):
            pre, query, post = sent_toks[:i], sent_toks[i], sent_toks[i+1:]
            window_toks = {i for i in pre[-args.keep_window_size:] + post[:args.keep_window_size]}
            results.append((window_toks, pre, query, post))
    return results, doc['meshMajor'], doc['journal']


def main(args):
    keep_vocab = {line.strip() for line in open(args.keep_vocab)}
    all_vocab = {line.split(maxsplit=1)[0] for line in islice(open(args.all_vocab), 1, None)}
    unigrams = frozenset([i for i in all_vocab if len(i) == 1])

    total = int(subprocess.Popen(f'wc -l {args.input}', shell=True, stdout=subprocess.PIPE).stdout.read().split()[0])

    # get bioasq filters
    min_year = 2000
    min_counts = 10_000
    journal_data = defaultdict(list)
    logger.info(f'counting journal occurrences with year >= {min_year} and counts >= {min_counts}')
    for line in tqdm(islice(open(args.input), 1, None), total=total):
        doc = json.loads(line[:-1] if line[-1] == ',' else line[:-2])
        if len(doc["journal"].split()) == 1 and doc["year"] is not None and int(doc["year"]) >= min_year:
            journal_data[doc["journal"]].append(doc)
    journal_counts = {journal: len(docs) for journal, docs in journal_data.items()}
    valid_journals = [journal for journal, counts in journal_counts.items() if counts >= min_counts]
    logger.info(f'valid journals ({len(valid_journals)}): {sorted(valid_journals)}')
    valid_journal_data = [j for i in [docs for journal, docs in journal_data.items() if journal in valid_journals] for j in i]

    tok_journal_contexts = defaultdict(list)
    tok_journal_classes = defaultdict(lambda: Counter())

    with Pool(40) as pool:
        docs = pool.imap(process_bioasq, valid_journal_data)

        logger.info('processing json')
        for doc in tqdm(docs, total=len(valid_journal_data)):

            (results, tags, journal) = doc
            for (window_toks, pre, tok, post) in results:
                valid_tok = False if {i for i in tok}.difference(unigrams) else True
                if window_toks & keep_vocab and valid_tok:
                    tok_journal_contexts[tok].append((journal, pre, post))
                    tok_journal_classes[tok].update([journal])

    Example = namedtuple('Example', ['tag_frac', 'tags', 'tok', 'pre', 'post'])

    # PROCESS JOURNALS
    print('processing journal contexts..')
    tag_examples = defaultdict(list)
    for tok, journal_counts in tqdm(tok_journal_classes.items(), total=len(tok_journal_classes)):
        valid_journal_counts = {journal: count for journal, count in journal_counts.items() if journal in valid_journals}
        tok_tag = sorted(valid_journal_counts.keys(), key=lambda x: -valid_journal_counts[x])[0]
        tok_journal_frac = valid_journal_counts[tok_tag] / sum(valid_journal_counts.values())

        if tok_journal_frac < 0.5:
            continue
        result_examples = [Example(tok_journal_frac, tok_tag, tok, pre, post)
                           for (context_tag, pre, post) in tok_journal_contexts[tok] if tok_tag == context_tag]
        assert result_examples, f'{tok}, {tok_journal_frac}, {valid_journal_counts}, {tok_journal_contexts[tok]}'

        if tok not in keep_vocab and tok in all_vocab:  # testing vocab only
            tag_examples[tok_tag].extend(result_examples)

    # PROCESS JOURNALS
    class_frac_limit = 2/3  # use supermajority for classification
    # for class_frac_type, class_frac_limit in [('maj', 1/2), ('supermaj', 2/3)]:
    output_prefix = f'{args.out_dir}/{args.keep_vocab.stem}/{args.input.stem}_filtered_win_{args.keep_window_size}_intersect_vocab'

    # output all
    Path(output_prefix).parent.mkdir(exist_ok=True, parents=True)

    with open(f'{output_prefix}_classes.all', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'tok'], quoting=csv.QUOTE_ALL)
        for tag, examples in tag_examples.items():
            toks = {e.tok for e in examples}
            output = [{'tag': tag, 'tok': tok} for tok in toks]
            writer.writerows(output)

    output = {tag: [{'tag': tag,
                     'tok': e.tok,
                     'pre': ' '.join(e.pre),
                     'post': ' '.join(e.post)}
                    for e in examples if e.tag_frac >= class_frac_limit]
              for tag, examples in tag_examples.items()}

    result_filename = f'{output_prefix}.all'  # might contain duplicates!
    with open(result_filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'tok', 'pre', 'post'], quoting=csv.QUOTE_ALL, extrasaction='ignore')
        for tag in output.keys():
            writer.writerows(output[tag])

    # output db (deduped & balanced)
    output_examples(f'{output_prefix}_db', tag_examples, dedupe=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=Path, default=Path('raw/bioASQ/allMeSH_2019.json'))
    argparser.add_argument('--out_dir', type=Path, default=Path('bioASQ'))
    argparser.add_argument('--all_vocab', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked-vocab.txt'))
    argparser.add_argument('--keep_vocab', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked_0.1.wp'))
    argparser.add_argument('--keep_window_size', type=int, default=7)
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)

    logger.info('Done!')
