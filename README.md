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
Once the probing results have been obtained, it is possible to compute the intersections of the top-100 units independently selected for each language. To do so the R script is in `intersections/super_exact_test.R`. Note that to successfully run the test the probing results must already have been run. The results are quite dense and the raw output of the R test is quite difficult to read. So, the script `read_intersect.py` helps in navigating the output.

> [!NOTE]
> In our study, we make use of several pre-existing affective ratings datasets. Remember to cite the relevant articles if you use these norms.

|Language | Family | Items | Raters |
|---|---|---|---|
|Montefinese et al. (2013) | Italian | ine  | 1,121 | 684 | 
|Warriner et al. (2013) | English | ine | 13,915 | 1,827 |
|Moors et al. (2013) | Dutch | ine | 4,300 | 224 |
|Redondo et al. (2007) | Spanish | ine | 1,034 | 720 |
|Imbir (2016) | Polish | ine | 4,900 | 400 |
|Ćoso et al. (2019) | Croatian | ine | 3,022 | 933 |
|Monnier and Syssau (2014) | French | ine | 1,031 | 469 |
|Palogiannidi et al. (2016) | Greek | ine | 1,034 | 105 |
|Schmidtke et al. (2014) | German | ine | 1,003 | 65 |
|Soares et al. (2011) | Portuguese | ine | 1,034 | 958 |
|Kapucu et al. (2021) | Turkish | trk | 2,031 | 1,527 |
|Yao et al. (2017) | Chinese | sit | 1,100 | 960 |
|Sianipar et al. (2016) | Indonesian | map | 1,402 | 1,490 |
