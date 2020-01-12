"""Converts MS MARCO data into TF Records that will be consumed by duoBERT."""
import collections
import json
import os
import tensorflow as tf
import time
import tokenization

from tqdm import tqdm


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_folder', None, 'Folder where the TFRecord files will be writen.')

flags.DEFINE_string(
    'vocab_file', None,
    'The vocabulary file that the BERT model was trained on.')

flags.DEFINE_string(
    'triples_train', None,
    'TSV file containing query, relevant and non-relevant docs.')

flags.DEFINE_string(
    'corpus', None, 'Path to the tsv file containing the paragraphs.')

flags.DEFINE_string(
    'queries_dev', None, 'Path to the <query id; query text> pairs for dev.')

flags.DEFINE_string(
    'queries_test', None,
    'Path to the <query id; query text> pairs for test.')

flags.DEFINE_string(
    'run_dev', None, 'Path to the query id / candidate doc ids pairs for dev.')

flags.DEFINE_string(
    'run_test', None,
    'Path to the query id / candidate doc ids pairs for test.')

flags.DEFINE_string(
    'qrels_dev', None,
    'Path to the query id / relevant doc ids pairs for dev.')

flags.DEFINE_integer(
    'num_dev_docs', 1000,
    'The number of docs per query for the development set.')

flags.DEFINE_integer(
    'num_test_docs', 1000,
    'The number of docs per query for the test set.')

flags.DEFINE_integer(
    'max_seq_length', 512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter than '
    'this will be padded.')

flags.DEFINE_integer(
    'max_query_length', 64,
    'The maximum query sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated.')

flags.DEFINE_string(
    'pad_doc_id', '5500000',
    'ID of the pad document. This pad document is added to the TF Records '
    'whenever the number of retrieved documents is lower than num_eval_docs. '
    'This can be any valid doc id.')


def convert_train(tokenizer):
    """Convert triples train to a TF Record file."""
    start_time = time.time()

    print('Counting the number of training examples...')
    num_examples = sum(1 for _ in open(FLAGS.triples_train))

    print('Converting to tfrecord...')
    with tf.python_io.TFRecordWriter(
            FLAGS.output_folder + '/dataset_train.tf') as writer:
        for i, line in tqdm(enumerate(open(FLAGS.triples_train)),
                            total=num_examples):
            query, relevant_doc, non_relevant_doc = line.rstrip().split('\t')

            query = tokenization.convert_to_unicode(query)
            query_ids = tokenization.convert_to_bert_input(
                text=query,
                max_seq_length=FLAGS.max_query_length,
                tokenizer=tokenizer,
                add_cls=True)

            labels = [1, 0]

            if i % 1000 == 0:
                print(f'query: {query}')
                print(f'Relevant doc: {relevant_doc}')
                print(f'Non-Relevant doc: {non_relevant_doc}\n')

            doc_token_ids = [
                tokenization.convert_to_bert_input(
                    text=tokenization.convert_to_unicode(doc_text),
                    max_seq_length=(
                        FLAGS.max_seq_length - len(query_ids)) // 2,
                    tokenizer=tokenizer,
                    add_cls=False)
                for doc_text in [relevant_doc, non_relevant_doc]
            ]

            input_ids = [
                query_ids + doc_token_ids[0] + doc_token_ids[1],
                query_ids + doc_token_ids[1] + doc_token_ids[0]
            ]
            segment_ids = [
                ([0] * len(query_ids) + [1] * len(doc_token_ids[0]) +
                    [2] * len(doc_token_ids[1])),
                ([0] * len(query_ids) + [1] * len(doc_token_ids[1]) +
                    [2] * len(doc_token_ids[0]))
            ]

            for input_id, segment_id, label in zip(
                    input_ids, segment_ids, labels):

                input_id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=input_id))

                segment_id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=segment_id))

                labels_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))

                features = tf.train.Features(feature={
                        'input_ids': input_id_tf,
                        'segment_ids': segment_id_tf,
                        'label': labels_tf,
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())


def convert_dataset(data, corpus, set_name, max_docs, tokenizer):
    """Convert dev or test dataset to a TF Record file."""
    ids_file = open(
            FLAGS.output_folder + '/query_doc_ids_' + set_name + '.txt' , 'w')
    output_path = FLAGS.output_folder + '/dataset_' + set_name + '.tf'

    print(f'Converting {set_name} to tfrecord')
    start_time = time.time()

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for i, query_id in tqdm(enumerate(data), total=len(data)):
            query, qrels, doc_ids = data[query_id]

            query = tokenization.convert_to_unicode(query)
            query_ids = tokenization.convert_to_bert_input(
                    text=query,
                    max_seq_length=FLAGS.max_query_length,
                    tokenizer=tokenizer,
                    add_cls=True)

            doc_ids = doc_ids[:max_docs]

            # Add fake docs so we always have max_docs per query.
            doc_ids += max(0, max_docs - len(doc_ids)) * [FLAGS.pad_doc_id]

            labels = [
                    1 if doc_id in qrels else 0
                    for doc_id in doc_ids
            ]

            if i % 1000 == 0:
                print(f'query: {query}; len qrels: {len(qrels)}')
                print(f'sum labels: {sum(labels)}')
                for j, (label, doc_id) in enumerate(zip(labels, doc_ids)):
                    print(f'doc {j}, label {label}, id: {doc_id}\n'
                          f'{corpus[doc_id]}\n\n')
                print()

            doc_token_ids = [
                    tokenization.convert_to_bert_input(
                            text=tokenization.convert_to_unicode(
                                corpus[doc_id]),
                            max_seq_length=(
                                FLAGS.max_seq_length - len(query_ids)) // 2,
                            tokenizer=tokenizer,
                            add_cls=False)
                    for doc_id in doc_ids
            ]
            input_ids = []
            segment_ids = []
            pair_doc_ids = []
            labels_pair = []
            for num_a, (doc_id_a, doc_token_id_a, label_a) in enumerate(
                    zip(doc_ids, doc_token_ids, labels)):
                for num_b, (doc_id_b, doc_token_id_b) in enumerate(
                        zip(doc_ids, doc_token_ids)):
                    if num_a == num_b:
                        continue
                    input_ids.append(
                        query_ids + doc_token_id_a + doc_token_id_b)
                    segment_ids.append((
                            [0] * len(query_ids) +
                            [1] * len(doc_token_id_a) +
                            [2] * len(doc_token_id_b)))
                    pair_doc_ids.append((doc_id_a, doc_id_b))
                    labels_pair.append(label_a)

            for input_id, segment_id, label, pair_doc_id in zip(
                    input_ids, segment_ids, labels_pair, pair_doc_ids):

                ids_file.write(
                    f'{query_id}\t{pair_doc_id[0]}\t{pair_doc_id[1]}\n')

                input_id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=input_id))

                segment_id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=segment_id))

                labels_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))

                features = tf.train.Features(feature={
                        'input_ids': input_id_tf,
                        'segment_ids': segment_id_tf,
                        'label': labels_tf,
                })

                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    ids_file.close()


def load_qrels(path):
    """Loads qrels into a dict of key: query_id, value: list of relevant doc
    ids."""
    qrels = collections.defaultdict(set)
    print(f'Loading qrels: {path}')
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_id, relevance = line.rstrip().split('\t')
            if int(relevance) >= 1:
                qrels[query_id].add(doc_id)
    return qrels


def load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    queries = {}
    print(f'Loading queries: {path}')
    with open(path) as f:
        for line in tqdm(f):
            query_id, query = line.rstrip().split('\t')
            queries[query_id] = query
    return queries


def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    run = collections.OrderedDict()
    print(f'Loading run: {path}')
    with open(path) as f:
        for line in tqdm(f):
            query_id, doc_id, rank= line.split('\t')
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_id, int(rank)))


    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_ids_ranks in run.items():
        sorted(doc_ids_ranks, key=lambda x: x[1])
        doc_ids = [doc_ids for doc_ids, _ in doc_ids_ranks]
        sorted_run[query_id] = doc_ids

    return sorted_run


def merge(qrels, run, queries):
    """Merge qrels and runs into a single dict of key: query,
    value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for query_id, candidate_doc_ids in run.items():
        query = queries[query_id]
        relevant_doc_ids = set()
        if qrels:
            relevant_doc_ids = qrels[query_id]
        data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
    return data


def load_corpus(path):
    """Load corpus into a dictionary with keys as doc ids and values as doc
    texts."""
    corpus = {}
    with open(path) as f:
        for line in tqdm(f):
            doc_id, doc_text = line.strip().split('\t')
            corpus[doc_id] = doc_text
    return corpus


def main():
    if not os.path.exists(FLAGS.output_folder):
        os.makedirs(FLAGS.output_folder)

    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=True)

    print('Loading Corpus...')
    corpus = load_corpus(FLAGS.corpus)

    print('Converting Training Set...')
    convert_train(tokenizer=tokenizer)

    for set_name, queries_path, qrels_path, run_path, max_docs in [
            ('dev', FLAGS.queries_dev, FLAGS.qrels_dev, FLAGS.run_dev,
             FLAGS.num_dev_docs),
            ('test', FLAGS.queries_test, None, FLAGS.run_test,
             FLAGS.num_test_docs)]:

        print(f'Converting {set_name}')
        qrels = None
        if set_name != 'test':
            qrels = load_qrels(path=qrels_path)

        queries = load_queries(queries_path)
        run = load_run(path=run_path)
        data = merge(qrels=qrels, run=run, queries=queries)

        convert_dataset(data=data,
                        corpus=corpus,
                        set_name=set_name,
                        max_docs=max_docs,
                        tokenizer=tokenizer)
    print('Done!')


if __name__ == '__main__':
    main()
