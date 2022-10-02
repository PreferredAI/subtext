#! /bin/bash

mkdir -p raw
# shellcheck disable=SC2164
cd raw

url_ARXIV="https://github.com/NeelShah18/arxivData/raw/master/arxivData.json"
echo "Downloading arXiv dataset from: ${url_ARXIV}.."
mkdir -p "arXiv/"
wget -q -nc "${url_ARXIV}" -O "arXiv/arxivData.json"

url_20NEWS="http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
echo "Downloading 20News dataset from: ${url_20NEWS}.."
mkdir -p "20news-bydate/"
# combine train and test splits as we do not use them
wget -q -nc "${url_20NEWS}" -O - | tar -xzf - --strip-components 1 -C "20news-bydate/"

url_CLINC150="https://github.com/clinc/oos-eval/raw/master/data/data_full.json"
echo "Retrieving CLINC150 dataset from: ${url_CLINC150}.."
mkdir "clinc150/"
wget -q -nc ${url_CLINC150} -O "clinc150/data_full.json"

echo "Done! Saved to: $(pwd)"
