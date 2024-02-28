# affective-interlingua
Code to reproduce the paper "The Emergence of Semantic Units in Massively Multilingual Models" (LREC-COLING 2024).

## Probing
The main code to reproduce the cross-lingual probing experiments is in the folder `/probing`. 

The folder also contains the necessary data files:
* Sentences (e.g., `chinese_sentences_train`
* Affective variables (e.g., `chinese_arousal_train`)
* Indexes of the target word in the sentences (e.g., `chinese_idx_train`)

The two scripts of interest are `xlmr.py` and `mbert.py`. They should be run as:

```
python3 xlmr.py -c [NAME_OF_THE_LANGUAGE]
```
