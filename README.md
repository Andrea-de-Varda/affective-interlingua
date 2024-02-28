# Semantic units in multilingual models
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
The folder also contains the script `test_in_finnish.py` which tests whether the neurons identified in the main experiment are predictive of valence and arousal in a held-out language belonging to a language family that did not concur to identify the top neurons (Finnish).

The script makes heavy use of the package `NeuroX`. Make sure to [check it out](https://neurox.qcri.org/).

## Testing intersections
Once the probing results have been obtained, it is possible to compute the intersections of the top-100 units independently selected for each language. To do so the R script is in `intersections/super_exact_test.R`. Note that to successfully run the test the probing results must already have been run.
