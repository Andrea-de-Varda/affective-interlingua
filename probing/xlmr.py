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
sys.path.append("..")

# hyperparameters
l1 = 0.001
l2 = 0.001
epochs = 500
l_rate = 0.00001

parser = argparse.ArgumentParser(description='Probing on XLM-R - affective variables')
parser.add_argument('-c_words', action='store', dest='words', help='Store word corpus destination')
results = parser.parse_args()

dirName = results.words+"_XLM-R"
try:
    mkdir(dirName)
    print("Directory" , dirName ,  "created") 
except FileExistsError:
    print("Directory" , dirName ,  "already exists")

file_train_sentences = results.words+"_sentences_train"
file_test_sentences = results.words+"_sentences_test"
file_train_idx = results.words+"_idx_train"
file_test_idx = results.words+"_idx_test"
file_train_val = results.words+"_valence_train"
file_test_val = results.words+"_valence_test"
file_train_ar = results.words+"_arousal_train"
file_test_ar = results.words+"_arousal_test"

##############################
# EXTRACTING REPRESENTATIONS #
##############################

import neurox.data.extraction.transformers_extractor as transformers_extractor

if path.isfile(dirName+'/activations_train.json') == False:
    transformers_extractor.extract_representations('xlm-roberta-base',
        file_train_sentences,
        dirName+'/activations_train.json',
        aggregation="average"
    )

if path.isfile(dirName+'/activations_test.json') == False:
    transformers_extractor.extract_representations('xlm-roberta-base',
        file_test_sentences,
        dirName+'/activations_test.json',
        aggregation="average"
    )

import neurox.data.loader as data_loader
activations_train, num_layers_train = data_loader.load_activations(dirName+'/activations_train.json', 768, is_brnn=True)
activations_test, num_layers_test = data_loader.load_activations(dirName+'/activations_test.json', 768, is_brnn=True)

# load token inputs
source_train = [[item] for item in open(file_train_sentences).read().split("\n")[:-1] if item]
source_test = [[item] for item in open(file_test_sentences).read().split("\n")[:-1] if item]

# checking that array shape matches sentence length
for array, sentence in zip(activations_train, source_train):
    sentence = sentence[0].split()
    if len(array) != len(sentence):
        print("Error: array shape does not match sentence length")
for array, sentence in zip(activations_test, source_test):
    sentence = sentence[0].split()
    if len(array) != len(sentence):
        print("Error: array shape does not match sentence length")

# only extract representation in the verb position
positions_train = loadtxt(file_train_idx, delimiter="\n", unpack=False).astype(int)
positions_test = loadtxt(file_test_idx, delimiter="\n", unpack=False).astype(int)

# normalize to get Betas as weights
def normalize_activation(act):
    activations_out = []
    sd = np.std(act, axis=0)
    m = np.mean(act, axis=0)
    for activation in act:
        act_norm = (activation-m)/sd
        activations_out.append(act_norm)
    return activations_out

act_train = []
for array, index in zip(activations_train, positions_train):
    v = array[index]
    act_train.append(v.reshape(1, 9984))
activations_train = normalize_activation(act_train)

act_test = []
for array, index in zip(activations_test, positions_test):
    v = array[index]
    act_test.append(v.reshape(1, 9984))
activations_test = normalize_activation(act_test)

# load output vecs
valence_float_train = loadtxt(file_train_val, delimiter="\n", unpack=False).astype(float)
valence_float_test = loadtxt(file_test_val, delimiter="\n", unpack=False).astype(float)

ar_float_train = loadtxt(file_train_ar, delimiter="\n", unpack=False).astype(float)
ar_float_test = loadtxt(file_test_ar, delimiter="\n", unpack=False).astype(float)

# checking shape
print("Check shape train =", len(source_train) == len(valence_float_train) == len(ar_float_train) == len(activations_train))
print("Check shape test =", len(source_test) == len(valence_float_test) == len(ar_float_test) == len(activations_test))

tokens_train_val = {'source': source_train, 'target': [[i] for i in valence_float_train]}
tokens_test_val = {'source': source_test, 'target': [[i] for i in valence_float_test]}
tokens_train_ar = {'source': source_train, 'target': [[i] for i in ar_float_train]}
tokens_test_ar = {'source': source_test, 'target': [[i] for i in ar_float_test]}

###########
# VALENCE # ###########################
###########

import neurox.interpretation.utils as utils

X_train_val, y_train_val, mapping_train_val = utils.create_tensors(tokens_train_val, activations_train, task_specific_tag=None, task_type= "regression")
src2idx_train_val, idx2src_train_val = mapping_train_val

X_test_val, y_test_val, mapping_test_val = utils.create_tensors(tokens_test_val, activations_test, task_specific_tag=None, task_type= "regression")
src2idx_test_val, idx2src_test_val = mapping_test_val

############################
# TRAIN PROBING CLASSIFIER #
############################

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def regression_metrics(predictions, gold):
    MSE = mean_squared_error(gold, predictions)
    r2 = r2_score(gold, predictions)
    r = pearsonr(gold, predictions)
    return MSE, r2, r[0]

import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation

#################
# whole network #
#################

probe_val = linear_probe.train_linear_regression_probe(X_train_val, y_train_val, lambda_l1=l1, lambda_l2=l2, learning_rate = l_rate, num_epochs=epochs) # NOTE: at the time of scripting, there is a bug in the NeuroX code. "_train_probe() got an unexpected keyword argument "model_type". "model_type" instead of "task_type". I corrected the code in the relevant package
probe_eval_val, predictions_val = linear_probe.evaluate_probe(probe_val, X_test_val, y_test_val, return_predictions=True)
preds_val = [i[1][0] for i in predictions_val]

metrics_val = regression_metrics(preds_val, valence_float_test); print(metrics_val)

# saving whole network results 
with open(dirName+'/RESULTS_whole_network_valence.txt', 'w', encoding='utf-8') as f:
    f.write(f"MSE =  {metrics_val[0]}, R2 =  {metrics_val[1]}, r =  {metrics_val[2]}")

weights = list(probe_val.parameters())[0].data.cpu().numpy()
weights_abs = [n for n in np.abs(weights)]
orderings = [n for n in weights[0]]
orderings_abs = [n for n in weights_abs[0]]

neuron_ordering = pd.DataFrame(enumerate([n for n in weights_abs[0]]), columns=["neuron", "absolute_weight"])
neuron_ordering = neuron_ordering.sort_values(by="absolute_weight", ascending=False)
neuron_weights = pd.DataFrame(enumerate([n for n in weights[0]]), columns=["neuron", "weight"])
neuron_ordering.to_csv(dirName+"/whole_network_valence_ordering_normalized.csv", index=False)
neuron_weights.to_csv(dirName+"/whole_network_valence_raw_weights_normalized.csv", index=False)

########################
## layer-wise probing ##
########################

layer_dict = {layer : [neuron + layer*768 for neuron in range(768)] for layer in range(13)}

layerwise = []
for l in range(0, 13):
    layer_test = ablation.zero_out_activations_keep_neurons(X_test_val, layer_dict[l])
    out, predictions = linear_probe.evaluate_probe(probe_val, layer_test, y_test_val, return_predictions=True)
    preds = [i[1][0] for i in predictions]
    metrics = regression_metrics(preds, valence_float_test)
    layerwise.append([l, metrics[0], metrics[1], metrics[2]])
       
layerwise = pd.DataFrame(layerwise, columns=["l", "MSE", "R2", "r"])
print(layerwise)
layerwise.to_csv(dirName+"/layerwise_valence_normalized.csv", index=False)

###########
# AROUSAL # ###########################
###########

X_train_ar, y_train_ar, mapping_train_ar = utils.create_tensors(tokens_train_ar, activations_train, task_specific_tag=None, task_type= "regression")
src2idx_train_val, idx2src_train_val = mapping_train_ar

X_test_ar, y_test_ar, mapping_test_ar = utils.create_tensors(tokens_test_ar, activations_test, task_specific_tag=None, task_type= "regression")
src2idx_test_ar, idx2src_test_ar = mapping_test_ar

############################
# TRAIN PROBING CLASSIFIER #
############################

#################
# whole network #
#################

probe_ar = linear_probe.train_linear_regression_probe(X_train_ar, y_train_ar, lambda_l1=l1, lambda_l2=l2, learning_rate = l_rate, num_epochs=epochs)
probe_eval_ar, predictions_ar = linear_probe.evaluate_probe(probe_ar, X_test_ar, y_test_ar, return_predictions=True)
preds_ar = [i[1][0] for i in predictions_ar]

metrics_ar = regression_metrics(preds_ar, ar_float_test); print(metrics_ar)

with open(dirName+'/RESULTS_whole_network_arousal.txt', 'w', encoding='utf-8') as f:
    f.write(f"MSE =  {metrics_ar[0]}, R2 =  {metrics_ar[1]}, r =  {metrics_ar[2]}")


weights_ar = list(probe_ar.parameters())[0].data.cpu().numpy()
weights_abs_ar = [n for n in np.abs(weights_ar)]
orderings_ar = [n for n in weights_ar[0]]
orderings_abs_ar = [n for n in weights_abs_ar[0]]

neuron_ordering_ar = pd.DataFrame(enumerate([n for n in weights_abs_ar[0]]), columns=["neuron", "absolute_weight"])
neuron_ordering_ar = neuron_ordering_ar.sort_values(by="absolute_weight", ascending=False)
neuron_weights_ar = pd.DataFrame(enumerate([n for n in weights_ar[0]]), columns=["neuron", "weight"])
neuron_ordering_ar.to_csv(dirName+"/whole_network_arousal_ordering_normalized.csv", index=False)
neuron_weights_ar.to_csv(dirName+"/whole_network_arousal_raw_weights_normalized.csv", index=False)

######################
# layer-wise probing #
######################

layerwise = []
for l in range(0, 13):
    layer_test = ablation.zero_out_activations_keep_neurons(X_test_ar, layer_dict[l])
    out, predictions = linear_probe.evaluate_probe(probe_ar, layer_test, y_test_ar, return_predictions=True)
    preds = [i[1][0] for i in predictions]
    metrics = regression_metrics(preds, ar_float_test)
    layerwise.append([l, metrics[0], metrics[1], metrics[2]])
       
layerwise = pd.DataFrame(layerwise, columns=["l", "MSE", "R2", "r"])
print(layerwise)
layerwise.to_csv(dirName+"/layerwise_arousal_normalized.csv", index=False)
