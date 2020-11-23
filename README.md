# duoBERT

duoBERT is a pairwise ranking model based on BERT that is the last stage of a multi-stage retrieval pipeline:

![duobert](duobert_architecture.svg)

To train and re-rank with monoBERT, please check [this repository](https://github.com/nyu-dl/dl4marco-bert).

As of Jan 13th 2020, our MS MARCO leaderboard entry is the top scoring model with available code:

MSMARCO Passage Re-Ranking Leaderboard (Jan 13th 2020) | Eval MRR@10  | Dev MRR@10
------------------------------------- | :------: | :------:
SOTA - Enriched BERT base + AOA index + CAS | 0.393 | 0.408
BM25 + monoBERT + duoBERT + TCP (this code) | 0.379 | 0.390

For more details, check out our paper:

+ Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. [Multi-Stage Document Ranking with BERT.](https://arxiv.org/abs/1910.14424) _arXiv:1910.14424_, October 2019.

**NOTE!** The duoBERT model is no longer under active development and this repo is no longer being maintained.
We have shifted our efforts to [ranking with sequence-to-sequence models](https://www.aclweb.org/anthology/2020.findings-emnlp.63/).
A T5-based variant of the mono/duo design is described in [an overview of our submissions to the TREC-COVID challenge](https://www.aclweb.org/anthology/2020.sdp-1.5/), and a more detailed description of mono/duoT5 is in preparation.

## Data and Trained Models

We make the following data available for download:

+ `bert-large-msmarco-pretrained_only.zip`: monoBERT large pretrained on the MS MARCO corpus but not finetuned on the ranking task. We pretrained this model starting from the original BERT-large WWM (Whole Word Mask) checkpoint. It was pretrained for 100k iterations, batch size 128, learning rate 3e-6, and 10k warmup steps. We finetuned monoBERT and duoBERT from this checkpoint.
+ `monobert-large-msmarco-pretrained-and-finetuned.zip`: monoBERT large pretrained on the MS MARCO corpus and finetuned on the MS MARCO ranking task.
+ `duobert-large-msmarco-pretrained-and-finetuned.zip`: duoBERT large pretrained on the MS MARCO corpus and finetuned on the MS MARCO ranking task.
+ `run.bm25.dev.small.tsv`:  Approximately 6,980,000 pairs of dev set queries and retrieved passages using BM25. In this tsv file, the first column is the query id, the second column is the passage id, and the third column is the rank of the passage. There are 1000 passages per query in this file.
+ `run.bm25.test.small.tsv`:  Approximately 6,837,000 pairs of test set queries and retrieved passages using BM25.
+ `run.monobert.dev.small.tsv`:  Approximately 6,980,000 pairs of dev set queries and retrieved passages using BM25 and re-ranked with monoBERT. In this tsv file, the first column is the query id, the second column is the passage id, and the third column is the rank of the passage. There are 1000 passages per query in this file.
+ `run.monobert.test.small.tsv`:  Approximately 6,837,000 pairs of test set queries and retrieved passages using BM25 and re-ranked with monoBERT.
+ `run.duobert.dev.small.tsv`:  Approximately 6,980 x 30 pairs of dev set queries and passages re-ranked using duoBERT. In this run, the input to duoBERT were the top-30 passages re-ranked by monoBERT.
+ `run.duobert.test.tsv`:  Approximately 6,837 x 30 pairs of test set queries and passages re-ranked using duoBERT. In this run, the input to duoBERT were the top-30 passages re-ranked by monoBERT.
+ `dataset_train.tf`:  Approximately 80M pairs of training set queries and passages (40M relevant and 40M non-relevant) in the TF Record format.
+ `dataset_dev.tf`:  Approximately 6,980 x 30 pairs of dev set queries and passages in the TF Record format. These top-30 passages will be re-ranked by duoBERT.
+ `dataset_test.tf`:  Approximately 6,837 x 30 pairs of test set queries and passages in the TF Record format. These top-30 passages will be re-ranked by duoBERT.
+ `query_doc_ids_dev.txt`:  Approximately 6,980 x 30 pairs of query and doc id that will be used during inference.
+ `query_doc_ids_test.txt`:  Approximately 6,837 x 30 pairs of query and doc id that will be used during inference.
+ `queries.dev.small.tsv`: 6,980 queries from the MS MARCO dev set. In this tsv file, the first column is the query id, and the second is the query text.
+ `queries.eval.small.tsv`: 6,837 queries from the MS MARCO test (eval) set. In this tsv file, the first column is the query id, and the second is the query text.
+ `qrels.dev.small.tsv`: 7,437 pairs of query relevant passage ids from the MS MARCO dev set. In this tsv file, the first column is the query id, and the third column is the passage id. The other two columns (second and fourth) are not used.
+ `collection.tar.gz`: All passages (8,841,823) in the MS MARCO passage corpus. In this tsv file, the first column is the passage id, and the second is the passage text.
+ `triples.train.small.tar.gz`: Approximatelly 40M triples of query, relevant and non-relevant passages that are used to train duoBERT.

Download and verify the above files from the below table:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`bert-large-msmarco-pretrained-only.zip` |  3.44 GB | `88f1d0bd351058b1da1eb49b60c2e750` | [[Dropbox](https://www.dropbox.com/s/nvqs8qk7q63qr0s/bert-large-msmarco-pretrained-only.zip?dl=1)]
`monobert-large-msmarco-pretrained-and-finetuned.zip` | 3.42 GB | `db201b6433b3e605201746bda6b7723b` | [[Dropbox](https://www.dropbox.com/s/fhy7vf5488muz9u/monobert-large-msmarco-pretrained-and-finetuned.zip?dl=1)]
`duobert-large-msmarco-pretrained-and-finetuned.zip` | 3.43 GB | `dcae7441103ae8241f16df743b75337b` | [[Dropbox](https://www.dropbox.com/s/kxd8fitk4ax1hb5/duobert-large-msmarco-pretrained-and-finetuned.zip?dl=1)]
`run.bm25.dev.small.tsv.gz` | 44 MB | `0a7802ab41999161339087186dda4145` | [[Dropbox](https://www.dropbox.com/s/5pqpcnlzlib2b3a/run.bm25.dev.small.tsv.gz?dl=1)]
`run.bm25.test.small.tsv.gz` | 43 MB | `1ea465405f6a2467cb62015454bc88c7` | [[Dropbox](https://www.dropbox.com/s/6fzxajh79dkw8s1/run.bm25.test.small.tsv.gz?dl=1)]
`run.monobert.dev.small.tsv.gz` | 44 MB | `dee6065e7177facb7c740f607e40ac63` | [[Dropbox](https://www.dropbox.com/s/h5kiff0ofn3djvf/run.monobert.dev.small.tsv.gz?dl=1)]
`run.monobert.test.small.tsv.gz` | 43 MB | `f0e16234351a0a81d83f188e72662fbd` | [[Dropbox](https://www.dropbox.com/s/ctccble07k7lvlc/run.monobert.test.small.tsv.gz?dl=1)]
`run.duobert.dev.small.tsv.gz` | 2.0 MB | `0be1f12ab7c7bd2d913d31756a8f0a19` | [[Dropbox](https://www.dropbox.com/s/fffu74voideid5p/run.duobert.dev.small.tsv.gz?dl=1)]
`run.duobert.test.small.tsv.gz` | 2.0 MB | `0d4f1770f8be20411ed8c00fb727103d` | [[Dropbox](https://www.dropbox.com/s/93bj0ehhse3fbuv/run.duobert.test.small.tsv.gz?dl=1)]
`dataset_train.tf.gz` | 8.8 GB | `7a3a6705f3662837a1e874d7ed970d27` | [[Dropbox](https://www.dropbox.com/s/zi46r0905d2y908/dataset_train.tf.gz?dl=1)]
`dataset_dev.tf.gz` | 241 MB | `f4966bd5426092564a59c1a1c8e34539` | [[Dropbox](https://www.dropbox.com/s/yykiop01sto1fzf/dataset_dev.tf.gz?dl=1)]
`dataset_test.tf.gz` | 236 MB | `5387a926950b112616926fe3d475a22f` | [[Dropbox](https://www.dropbox.com/s/qx97yhq34ndtc7p/dataset_test.tf.gz?dl=1)]
`query_doc_ids_dev.txt.gz` | 19 MB | `05361aead605c1b8a8cc8d71ef3ff0f8` | [[Dropbox](https://www.dropbox.com/s/ttml8v0irfsmqcv/query_doc_ids_dev.txt.gz?dl=1)]
`query_doc_ids_test.txt.gz` | 19 MB | `5e657dff1e1f0748d29b291e5c731f9f` | [[Dropbox](https://www.dropbox.com/s/jvtf3qa8ux3wma8/query_doc_ids_test.txt.gz?dl=1)]
`queries.dev.small.tsv` | 283 KB | `41e980d881317a4a323129d482e9f5e5` | [[Dropbox](https://www.dropbox.com/s/iyw98nof7omynst/queries.dev.small.tsv?dl=1)]
`queries.eval.small.tsv` | 274 KB | `bafaf0b9eb23503d2a5948709f34fc3a` | [[Dropbox](https://www.dropbox.com/s/yst2tz1s9i2z5mx/queries.eval.small.tsv?dl=1)]
`qrels.dev.small.tsv` | 140 KB| `38a80559a561707ac2ec0f150ecd1e8a` | [[Dropbox](https://www.dropbox.com/s/ie27l0mzcjb5fbc/qrels.dev.small.tsv?dl=1)]
`collection.tar.gz` | 987 MB | `87dd01826da3e2ad45447ba5af577628` | [[Dropbox](https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz?dl=1)]
`triples.train.small.tar.gz` | 7.4 GB | `c13bf99ff23ca691105ad12eab837f84` | [[Dropbox](https://www.dropbox.com/s/6r4a8hpcgq0szep/triples.train.small.tar.gz?dl=1)]

All of the above files are stored in [this repo](https://git.uwaterloo.ca/jimmylin/duobert-data).
As an alternative to downloading each file separately, clone the repo and you'll have everything.

## Replicating our MS MARCO results with duoBERT
Here we provide instructions on how to replicate our BM25 + monoBERT + duoBERT + TCP dev run on MS MARCO leaderboard.

NOTE 1: we will run these experiments using a TPU; thus, you will need a Google Cloud account. Alternatively, you can use a GPU, but we haven't tried ourselves.

NOTE 2: For instructions on how to train and run inference using monoBERT, please check this [repository](https://github.com/nyu-dl/dl4marco-bert).

First download the following files (using the links in the table above):
- `qrels.dev.small.tsv`
- `dataset_dev.tf`
- `duobert-large-msmarco-pretrained-and-finetuned.zip`

Unzip `duobert-large-msmarco-pretrained-and-finetuned.zip` and upload the files to a bucket in the Google Cloud Storage.

Create a virtual machine with TPU in the Google Cloud. We provide below a
command-line example that should be executed in the Google Cloud Shell (change `your-tpu` 
accordingly):
```
ctpu up --zone=us-central1-b --name your-tpu --tpu-size=v3-8 --disk-size-gb=250 \
  --machine-type=n1-standard-4 --preemptible --tf-version=1.15 --noconf
```

ssh into the virtual machine and clone the git repo:
```
git clone https://github.com/castorini/duobert.git
```

Run duoBERT in evaluation mode (change `your-tpu` and `your-bucket` accordingly):
```
python run_duobert_msmarco.py \
  --data_dir=gs://your-bucket \
  --bert_config_file=gs://your-bucket/bert_config.json \
  --output_dir=. \
  --init_checkpoint=gs://your-bucket/model.ckpt-100000 \
  --max_seq_length=512 \
  --do_train=False \
  --do_eval=True \
  --eval_batch_size=128 \
  --num_eval_docs=30 \
  --use_tpu=True \
  --tpu_name=your-tpu \
  --tpu_zone=us-central1-b
```

This inference takes approximately 4 hours on a TPU v3. 
Once finished, run the evaluation script:
```
python3 msmarco_eval.py qrels.dev.small.tsv ./msmarco_predictions_dev.tsv
```

The output should be like this:
```
#####################
MRR @10: 0.3904377586755809
QueriesRanked: 6980
#####################
```

## Training DuoBERT
Here we provide instructions to train duoBERT. Note that a fully trained model is available in the above table.

First download the following files (using the links in the table above):
- `qrels.dev.small.tsv`
- `dataset_train.tf`
- `bert-large-msmarco-pretrained-only.zip`

Unzip `bert-large-msmarco-pretrained-only.zip` and upload all files to your Google Cloud Storage bucket.

Run duoBERT in training mode (change `your-tpu` and `your-bucket` accordingly):
```
python run_duobert_msmarco.py \
  --data_dir=gs://your-bucket \
  --bert_config_file=gs://your-bucket/bert_config.json \
  --output_dir=gs://your-bucket/output \
  --init_checkpoint=gs://your-bucket/model.ckpt-100000 \
  --max_seq_length=512 \
  --do_train=True \
  --do_eval=False \
  --learning_rate=3e-6 \
  --train_batch_size=128 \
  --num_train_steps=100000 \
  --num_warmup_steps=10000 \
  --use_tpu=True \
  --tpu_name=your-tpu \
  --tpu_zone=us-central1-b
```

This training should take approximately 30 hours on a TPU v3.


## Creating a TF Record dataset
Here we provide instructions to create the training, dev, and test TF Record files that are consumed by duoBERT. Note that these files are available in the above table.

Use the links from the table above to download the following files:
- `collection.tar.gz` (needs to be uncompressed)
- `triples.train.small.tar.gz` (needs to be uncompressed)
- `queries.dev.small.tsv`
- `queries.eval.small.tsv`
- `run.monobert.dev.small.tsv`
- `run.monobert.test.small.tsv`
- `qrels.dev.small.tsv`
- `vocab.txt` (available in `duobert-large-msmarco-pretrained-and-finetuned.zip`)

```
python convert_msmarco_to_duobert_tfrecord.py \
  --output_folder=. \
  --corpus=collection.tsv \
  --vocab_file=vocab.txt \
  --triples_train=triples.train.small.tsv \
  --queries_dev=queries.dev.small.tsv \
  --queries_test=queries.eval.small.tsv \
  --run_dev=run.monobert.dev.small.tsv \
  --run_test=run.monobert.test.small.tsv \
  --qrels_dev=qrels.dev.small.tsv \
  --num_dev_docs=30 \
  --num_test_docs=30 \
  --max_seq_length=512 \
  --max_query_length=64
``` 

This conversion takes approximately 30-50 hours and will produce the following files:
- `dataset_train.tf`
- `dataset_dev.tf`
- `dataset_test.tf`
- `query_doc_ids_dev.txt`
- `query_doc_ids_test.txt`


## How do I cite this work?
```
@article{nogueira2019multi,
  title={Multi-stage document ranking with BERT},
  author={Nogueira, Rodrigo and Yang, Wei and Cho, Kyunghyun and Lin, Jimmy},
  journal={arXiv preprint arXiv:1910.14424},
  year={2019}
}
```
