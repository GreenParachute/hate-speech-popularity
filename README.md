# hate-speech-popularity
Repository of data and code used for a hate speech prediction study

This repository contains the following files:

- anonymized training dataset (16133 instances) with extracted tweet, user and content features (features obtainable from twitter API are not included)
- script for training hate speech and tweet popularity classifiers from extracted features (assumes availability of Twitter API features)
- script for extracting n-gram features and training a classifier based just on n-gram features (assumes you have tweet text available; we do not provide this)

To do:
- 


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
