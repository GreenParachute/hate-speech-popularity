# hate-speech-popularity
Repository of data and code used for a hate speech prediction study

This repository contains the following files:

- anonymized training dataset (16130 instances) with extracted tweet, user and content features (features obtainable from twitter API are not included)
- script for extracting n-gram features and training a hate speech and tweet popularity classifiers from extracted features and n-gram features 

Example use of script:

```
$ python hate_speech_prediction.py --input features.csv --features nlp --model svm

$ python hate_speech_prediction.py --input corpus.csv --features ngram --model reg
```

NOTE: 

1) popularity prediction requires availability of features obtained through Twitter API

2) n-gram extractor requires tweet text; we do not provide this

3) script only works with python 2.7


Feel free to contact us with any questions or concerns, we'll be happy to help. 


# Citation:

If you use any of this data, please cite the following paper:

```
@inproceedings{examining-2018,
  title={Examining a hate speech corpus for hate speech detection and popularity prediction},
  author={Filip Klubi\v{c}ka and Raquel Fern\'{a}ndez},
  booktitle ={Proceedings of 4REAL: 1st Workshop on Replicability and Reproducibility of Research Results in Science and Technology of Language},
  year={2018}
}
```
https://arxiv.org/pdf/1805.04661.pdf
