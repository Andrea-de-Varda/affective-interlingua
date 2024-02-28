import neurox
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from os import mkdir, path, chdir
import pickle
import sys
import torch
import time
from researchpy import corr_pair
import random
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("..")

file_sentences = "finnish_sentences"
file_idx = "finnish_idx"
file_val = "finnish_valence"
file_ar = "finnish_arousal"

dirName = "finnish"
try:
    mkdir(dirName)
    print("Directory" , dirName ,  "created") 
except FileExistsError:
    print("Directory" , dirName ,  "already exists")

##############################
# EXTRACTING REPRESENTATIONS #
##############################

import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

if path.isfile(dirName+'/activations_train.json') == False:
    transformers_extractor.extract_representations('bert-base-multilingual-cased',
        file_sentences,
        dirName+'/activations_train.json',
        aggregation="average"
    )
    
if path.isfile(dirName+'/activations_train_xlm.json') == False:
    transformers_extractor.extract_representations('xlm-roberta-base',
        file_sentences,
        dirName+'/activations_train_xlm.json',
        aggregation="average"
    )

activations_bert, num_layers_bert = data_loader.load_activations(dirName+'/activations_train.json', 768, is_brnn=True)
activations_xlm, num_layers_xlm = data_loader.load_activations(dirName+'/activations_train_xlm.json', 768, is_brnn=True)

# load token inputs
source = [[item] for item in open(file_sentences).read().split("\n")[:-1] if item]

# checking that array shape matches sentence length
for array, sentence in zip(activations_bert, source):
    sentence = sentence[0].split()
    if len(array) != len(sentence):
        print("Error: array shape does not match sentence length")
        
for array, sentence in zip(activations_xlm, source):
    sentence = sentence[0].split()
    if len(array) != len(sentence):
        print("Error: array shape does not match sentence length")

# only extract representation in the target position
positions = loadtxt(file_idx, delimiter="\n", unpack=False).astype(int)

# normalize to get Betas as weights
def normalize_activation(act):
    activations_out = []
    sd = np.std(act, axis=0)
    m = np.mean(act, axis=0)
    for activation in act:
        act_norm = (activation-m)/sd
        activations_out.append(act_norm)
    return activations_out

act_bert = []
for array, index in zip(activations_bert, positions):
    v = array[index]
    act_bert.append(v.reshape(1, 9984))
activations_bert = normalize_activation(act_bert)

act_xlm = []
for array, index in zip(activations_xlm, positions):
    v = array[index]
    act_xlm.append(v.reshape(1, 9984))
activations_xlm = normalize_activation(act_xlm)

# load output vecs
valence_float = loadtxt(file_val, delimiter="\n", unpack=False).astype(float)
ar_float = loadtxt(file_ar, delimiter="\n", unpack=False).astype(float)

# checking shape
print("Check shape train =", len(source) == len(valence_float) == len(ar_float) == len(activations_bert) == len(activations_xlm))

################################
# neurons to test individually #
################################
# 484, 473 in valence_XLM, 11 degree - layer 5
# 139 in XLM-Arousal, 10 degree - layer 6
# 702, 289, 701, 309 in mBERT, valence (7 degree) - layer 7
# 662 in mBERT, arousal (8 degree) - layer 7

layer_dict = {layer : [neuron + layer*768 for neuron in range(768)] for layer in range(13)}

def random_sample_from_layer(layern, to_exclude, n=100):
    min_, max_ = layer_dict[layern][0], layer_dict[layern][767]
    out = []
    while len(out) < n:
        a = random.randint(min_, max_)
        if a not in out:
            if a not in to_exclude:
                out.append(a)
    return out

def test_neurons(activations, layer, target_neurons, affective):
    baseline = random_sample_from_layer(layer, target_neurons)
    for neuron in target_neurons:
        print("\n\n NEURON", layer_dict[layer][0]+neuron)
        df_aff = []
        for aff, act in zip(affective, activations):
            df_aff.append([aff, act[0][layer_dict[layer][neuron]]])
        df_aff = pd.DataFrame(df_aff, columns=["aff", "neuron"])
        x = sm.add_constant(df_aff[["neuron"]])
        y = df_aff["aff"]
        model = sm.OLS(y, x).fit()
        print(model.summary())
    t_values = []
    b = []
    for neuron in baseline:
        #print(neuron)
        df_baseline = []
        for aff, act in zip(affective, activations):
            df_baseline.append([aff, act[0][neuron]])
        df_base = pd.DataFrame(df_baseline, columns=["aff", "neuron"])
        x = sm.add_constant(df_base[["neuron"]])
        y = df_base["aff"]
        model = sm.OLS(y, x).fit()
        t = model.tvalues.iloc[1]
        t_values.append(abs(t))
        b.append(abs(model.params[1]))
    print(np.mean(b), np.mean(t_values))
    

test_neurons(activations_xlm, 5, [484, 473], valence_float)
test_neurons(activations_xlm, 6, [139], ar_float)

test_neurons(activations_bert, 7, [702, 289, 701, 309], valence_float)#1
test_neurons(activations_bert, 7, [662], ar_float)


test_neurons(activations_xlm, 5, [484], valence_float)
test_neurons(activations_xlm, 6, [139], ar_float)
test_neurons(activations_bert, 7, [701], valence_float)
test_neurons(activations_bert, 7, [479], ar_float)
