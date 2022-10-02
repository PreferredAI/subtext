import argparse
import csv
import datetime
import logging
import random
import string
from collections import Counter, defaultdict, namedtuple
from itertools import islice
from pathlib import Path

from blingfire import text_to_sentences, text_to_words
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger('process_20news_sent')
logger.setLevel(logging.DEBUG)


def output_examples(file_prefix, all_examples, num_samples=5, dedupe=True):
    if dedupe:
        def _group_examples_by_tok(_examples):
            examples_by_token = defaultdict(list)
            [examples_by_token[e.tok].append(e) for e in _examples]
            return dict(examples_by_token)

        all_examples_by_tok = {tag: _group_examples_by_tok(examples) for tag, examples in all_examples.items()}

        tag_counts = Counter([example[0].tag for examples in all_examples_by_tok.values() for example in examples.values()])
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
            writer = csv.DictWriter(f, fieldnames=['tag', 'choices', 'pre', 'post', 'y'], quoting=csv.QUOTE_ALL)
            for tag in output_examples.keys():
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


def main(args):
    keep_vocab = {line.strip() for line in open(args.keep_vocab)}
    unigrams = frozenset({i for i in keep_vocab if len(i) == 1})
    all_vocab = {line.split(maxsplit=1)[0] for line in islice(open(args.all_vocab), 1, None)}

    results = defaultdict(list)

    # 20NEWS PROCESSING
    logger.debug('reading categories..')
    for category_name in tqdm(args.input.iterdir(), total=20):
        tag = category_name.name
        for input_f in category_name.iterdir():
            for sent in text_to_sentences(open(input_f, errors='backslashreplace').read().replace('\n', ' ')).split('\n'):
                sent = sent.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
                sent_toks = [i.lower() for i in text_to_words(sent).split()]
                for i in range(len(sent_toks)):
                    pre, query, post = sent_toks[:i], sent_toks[i], sent_toks[i+1:]
                    if {i for i in query} - unigrams:
                        logger.warning(f'tok failed unigram decode: {query}')
                        continue
                    window_toks = {tok for tok in pre[-args.keep_window_size:] + post[:args.keep_window_size]
                                   if tok in keep_vocab and {i for i in tok} < unigrams}
                    if window_toks:
                        results[query].append((tag, pre, post))

    Example = namedtuple('Example', ['tag', 'tok', 'pre', 'post'])

    tag_examples = defaultdict(list)
    for tok, result in results.items():
        if tok not in keep_vocab and tok in all_vocab:  # testing vocab only
            tok_tag = {r[0] for r in result}
            if len(tok_tag) > 1:
                continue
            tok_tag = tok_tag.pop()
            result_examples = [Example(tok_tag, tok, pre, post) for (_, pre, post) in result]
            tag_examples[tok_tag].extend(result_examples)

    output_prefix = f'{args.out_dir}/{args.keep_vocab.stem}/{args.input.stem}_filtered_win_{args.keep_window_size}_intersect_vocab'
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
                    for e in examples]
              for tag, examples in tag_examples.items()}

    result_filename = f'{output_prefix}.all'
    with open(result_filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['tag', 'tok', 'pre', 'post'], quoting=csv.QUOTE_ALL)
        for tag in output.keys():
            writer.writerows(output[tag])

    # output db (deduped & balanced)
    output_examples(f'{output_prefix}_db', tag_examples, dedupe=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=Path, default=Path('raw/20news-bydate'))
    argparser.add_argument('--out_dir', type=Path, default=Path('20news-bydate'))
    argparser.add_argument('--all_vocab', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked-vocab.txt'))
    argparser.add_argument('--keep_vocab', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked_0.1.wp'))
    argparser.add_argument('--keep_window_size', type=int, default=7)
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
