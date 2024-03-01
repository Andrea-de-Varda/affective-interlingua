import pandas as pd
from os import chdir
import re
import numpy as np
import sys
from researchpy import corr_pair
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from itertools import combinations
from researchpy import ttest
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def make_float(n):
    try:
        out = float(n)
    except ValueError:
        out = np.nan
    return out
    
def make_df(filename, save=False):
    file = [re.sub("\n", "", s) for s in open(filename+".txt").readlines()]; print(len(file))
    out1 = []
    for f in file[4:8195]:
        ff = re.sub(" & ", "&", f)
        ff = ff.split(" ")
        ff = [f for f in ff if f]
        ff[2] = int(ff[2])
        out1.append(ff)
    
    out2 = []
    for f in file[8196:]:
        ff = f.split(" ")
        ff = [make_float(f) for f in ff if f][1:5]
        out2.append(ff)
    
    if len(out1) != len(out2):
        raise ValueError("The sizes do not match")
        
    df = []
    for z1, z2 in zip(out1, out2):
        a = z1+z2
        df.append(a)
    df = pd.DataFrame(df, columns=["code", "lang", "degree", "observed", "expected", "FE", "p"])
    if save:
        df.to_csv(filename+".csv", index=False)
    return df

all_languages = set("German Dutch French Italian Spanish Turkish Chinese English Indonesian Polish Greek Croatian Portuguese".split())

def test_df(df, print_all = True):
    toprint = {}
    for degree in range(2, 13):
        l = df[df.degree == degree]
        l1 = l[l.FE > 0]
        print(degree, round(len(l1)/len(l), 10))
        if degree > 10:
            if len(l) > 0:
                toprint[degree] = [col.split("&") for col in l.lang.tolist()]
    if print_all:
        if toprint:
            for degree, intersections in toprint.items():
                print("\n\nFOUND", len(intersections), "INTERSECTIONS FOR DEGREE", degree)
                for inters in intersections:
                    print("\n", *inters)
                    diff = all_languages.difference(set(inters))
                    print("Missing", *diff)
            
arousal = make_df("superset_arousal_xlm_normalized")
valence = make_df("superset_valence_xlm_normalized")

arousal_b = make_df("superset_arousal_normalized")
valence_b = make_df("superset_valence_normalized")

arousal2 = arousal[arousal.degree == 2] # 2nd degree
valence2 = valence[valence.degree == 2]

arousal_b2 = arousal_b[arousal_b.degree == 2]
valence_b2 = valence_b[valence_b.degree == 2]

# percentage of 2nd degree intersection with FE > 1 or significant
len(arousal2[arousal2.p < .05])/len(arousal2[arousal2.p.notnull()])
len(valence2[valence2.p < .05])/len(valence2[valence2.p.notnull()])

len(arousal2[arousal2.FE > 1])/len(arousal2[arousal2.FE.notnull()])
len(valence2[valence2.FE > 1])/len(valence2[valence2.FE.notnull()])

len(arousal_b2[arousal_b2.p < .05])/len(arousal_b2[arousal_b2.p.notnull()])
len(valence_b2[valence_b2.p < .05])/len(valence_b2[valence_b2.p.notnull()])

len(arousal_b2[arousal_b2.FE > 1])/len(arousal_b2[arousal_b2.FE.notnull()])
len(valence_b2[valence_b2.FE > 1])/len(valence_b2[valence_b2.FE.notnull()])


def test_df2(df, consider="p", max_degree = 9, myfunc=set.difference):
    for degree in range(2, 13): 
        if consider == "p":
            temp = df[df.degree == degree]
            print(degree, len(temp[temp.p < .05]), len(temp), round(len(temp[temp.p < .05])/len(temp[temp.p.notnull()]), 5))
            if degree > max_degree:
                s = temp[temp.p < .05].lang
                s_p = temp[temp.p < .05].p
                s_obs = temp[temp.p < .05].observed
                myFe = temp[temp.p < .05].FE
                for l, p, obs, fe in zip(s, s_p, s_obs, myFe):
                    print(myfunc(all_languages, set(l.split("&"))), p, obs, fe)
        elif consider == "FE":
            temp = df[df.degree == degree]
            print(degree, len(temp[temp.FE > 1]),len(temp), round(len(temp[temp.FE > 1])/len(temp[temp.FE.notnull()]), 5))
            if degree > max_degree:
                s = temp[temp.FE > 1].lang
                s_p = temp[temp.FE > 1].p
                s_obs = temp[temp.FE > 1].observed
                for l, p, obs in zip(s, s_p, s_obs):
                    print(myfunc(all_languages, set(l.split("&"))), p, obs)

# find highest-degree intersections
# XLM
test_df2(valence, max_degree=10)
test_df2(arousal)

test_df2(valence, consider="FE")
test_df2(arousal, consider="FE")

# mBERT
test_df2(valence_b, max_degree=6, myfunc=set.intersection)
test_df2(arousal_b, max_degree=7, myfunc=set.intersection)

test_df2(valence_b, consider="FE")
test_df2(arousal_b, consider="FE")

########################################################
# correlate average intersection size with performance #
########################################################

def find_avg_int(df):
    temp = df[df.degree == 2]
    temp[['First','Second']] = temp.lang.str.split("&",expand=True,)
    out = {key:[] for key in set(temp.First).union(set(temp.Second))}
    for lang in out.keys():
        for index, row in temp.iterrows():
            if lang in [row["First"], row["Second"]]:
                out[lang].append(row.observed)
    for key, value in out.items():
        out[key] = np.mean(value)
    return out

valence_overlap_xlm = find_avg_int(valence)
arousal_overlap_xlm = find_avg_int(arousal)
valence_overlap_b = find_avg_int(valence_b)
arousal_overlap_b = find_avg_int(arousal_b)

# markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
# colors = ['r','g','b','c','m', 'y', 'k']

result_bert_arousal = []
for lang in ["italian", "spanish", "dutch", "french", "german", "portuguese", "turkish", "english", "greek", "chinese", "polish", "croatian", "indonesian"]:
    wholenet = float(open(lang+"/RESULTS_whole_network_arousal.txt").read().split(",")[2][7:])
    result_bert_arousal.append([lang, wholenet])
result_bert_arousal = pd.DataFrame(result_bert_arousal, columns=["lang", "r"])
result_bert_arousal = result_bert_arousal.sort_values(by="r", ascending=False)
result_bert_arousal["lang"] = result_bert_arousal["lang"].str.title()
result_bert_arousal["overlap"] = result_bert_arousal["lang"].map(arousal_overlap_b)
result_bert_arousal["model"] = "mBERT-arousal"
result_bert_arousal["m"] = "mBERT"
result_bert_arousal["a"] = "arousal"
corr_pair(result_bert_arousal[["r", "overlap"]])
#plt.scatter(result_bert_arousal["r"], result_bert_arousal["overlap"])


result_bert_valence = []
for lang in ["italian", "spanish", "dutch", "french", "german", "portuguese", "turkish", "english", "greek", "chinese", "polish", "croatian", "indonesian"]:
    wholenet = float(open(lang+"/RESULTS_whole_network_valence.txt").read().split(",")[2][7:])
    result_bert_valence.append([lang, wholenet])
result_bert_valence = pd.DataFrame(result_bert_valence, columns=["lang", "r"])
result_bert_valence = result_bert_valence.sort_values(by="r", ascending=False)
result_bert_valence["lang"] = result_bert_valence["lang"].str.title()
result_bert_valence["overlap"] = result_bert_valence["lang"].map(valence_overlap_b)
result_bert_valence["model"] = "mBERT-valence"
result_bert_valence["m"] = "mBERT"
result_bert_valence["a"] = "valence"

corr_pair(result_bert_valence[["r", "overlap"]])
#plt.scatter(result_bert_valence["r"], result_bert_valence["overlap"])

result_xlm_arousal = []
for lang in ["italian", "spanish", "dutch", "french", "german", "portuguese", "turkish", "english", "greek", "chinese", "polish", "croatian", "indonesian"]:
    wholenet = float(open(lang+"_XLM-R/RESULTS_whole_network_arousal.txt").read().split(",")[2][7:])
    result_xlm_arousal.append([lang, wholenet])
result_xlm_arousal = pd.DataFrame(result_xlm_arousal, columns=["lang", "r"])
result_xlm_arousal = result_xlm_arousal.sort_values(by="r", ascending=False)
result_xlm_arousal["lang"] = result_xlm_arousal["lang"].str.title()
result_xlm_arousal["overlap"] = result_xlm_arousal["lang"].map(arousal_overlap_xlm)
result_xlm_arousal["model"] = "XLM-R-arousal"
result_xlm_arousal["m"] = "XLM-R"
result_xlm_arousal["a"] = "arousal"
corr_pair(result_xlm_arousal[["r", "overlap"]])
#plt.scatter(result_xlm_arousal["r"], result_xlm_arousal["overlap"])

result_xlm_valence = []
for lang in ["italian", "spanish", "dutch", "french", "german", "portuguese", "turkish", "english", "greek", "chinese", "polish", "croatian", "indonesian"]:
    wholenet = float(open(lang+"_XLM-R/RESULTS_whole_network_valence.txt").read().split(",")[2][7:])
    result_xlm_valence.append([lang, wholenet])
result_xlm_valence = pd.DataFrame(result_xlm_valence, columns=["lang", "r"])
result_xlm_valence = result_xlm_valence.sort_values(by="r", ascending=False)
result_xlm_valence["lang"] = result_xlm_valence["lang"].str.title()
result_xlm_valence["overlap"] = result_xlm_valence["lang"].map(valence_overlap_xlm)
result_xlm_valence["model"] = "XLM-R-valence"
result_xlm_valence["m"] = "XLM-R"
result_xlm_valence["a"] = "valence"
corr_pair(result_xlm_valence[["r", "overlap"]])
#plt.scatter(result_xlm_valence["r"], result_xlm_valence["overlap"])


results_all = pd.concat([result_bert_valence, result_bert_arousal, result_xlm_valence, result_xlm_arousal])
corr_pair(results_all[["r", "overlap"]])

results_all["Mean pairwise overlap"] = results_all["overlap"]
results_all["R"] = results_all["r"]

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

matplotlib.rcParams.update({'font.size': 14})

# slopes by MODEL X VARIABLE
plt.figure(dpi=300)
sns.regplot(x=results_all[results_all.model=="mBERT-valence"]["Mean pairwise overlap"], y = results_all[results_all.model=="mBERT-valence"]["R"], marker="o", color="royalblue", label="mBERT, valence")
sns.regplot(x=results_all[results_all.model=="mBERT-arousal"]["Mean pairwise overlap"], y = results_all[results_all.model=="mBERT-arousal"]["R"], marker="o", color="orangered", label="mBERT, arousal")
sns.regplot(x=results_all[results_all.model=="XLM-R-valence"]["Mean pairwise overlap"], y = results_all[results_all.model=="XLM-R-valence"]["R"], marker=",", color="royalblue", label="XLM-R, valence")
sns.regplot(x=results_all[results_all.model=="XLM-R-arousal"]["Mean pairwise overlap"], y = results_all[results_all.model=="XLM-R-valence"]["R"], marker=",", color="orangered", label="XLM-R, arousal")#, line_kws={'linewidth':1.6})
plt.legend(fontsize=9.6, loc="lower right")
plt.show()

# slopes by model and variable
plt.figure(dpi=300)
# scatters
sns.regplot(x=results_all[results_all.model=="mBERT-valence"]["Mean pairwise overlap"], y = results_all[results_all.model=="mBERT-valence"]["R"], marker="o", color="royalblue", label="mBERT, valence", fit_reg=False)
sns.regplot(x=results_all[results_all.model=="mBERT-arousal"]["Mean pairwise overlap"], y = results_all[results_all.model=="mBERT-arousal"]["R"], marker="o", color="orangered", label="mBERT, arousal", fit_reg=False)
sns.regplot(x=results_all[results_all.model=="XLM-R-valence"]["Mean pairwise overlap"], y = results_all[results_all.model=="XLM-R-valence"]["R"], marker=",", color="royalblue", label="XLM-R, valence", fit_reg=False)
sns.regplot(x=results_all[results_all.model=="XLM-R-arousal"]["Mean pairwise overlap"], y = results_all[results_all.model=="XLM-R-valence"]["R"], marker=",", color="orangered", label="XLM-R, arousal", fit_reg=False)#, line_kws={'linewidth':1.6})
# slopes
sns.regplot(x=results_all[results_all.a=="valence"]["Mean pairwise overlap"], y = results_all[results_all.a=="valence"]["R"], marker="o", color="gray", label="Valence", scatter=False, line_kws={"linestyle": '-'}, ci = 90) # ci = 90
sns.regplot(x=results_all[results_all.a=="arousal"]["Mean pairwise overlap"], y = results_all[results_all.a=="arousal"]["R"], marker="o", color="gray", label="Arousal", scatter=False, ci = 90, line_kws={"linestyle": '--'})
sns.regplot(x=results_all[results_all.m=="mBERT"]["Mean pairwise overlap"], y = results_all[results_all.m=="mBERT"]["R"], marker="o", color="gray", label="mBERT", scatter=False, ci = 90, line_kws={"linestyle": ':'})
sns.regplot(x=results_all[results_all.m=="XLM-R"]["Mean pairwise overlap"], y = results_all[results_all.m=="XLM-R"]["R"], marker="o", color="gray", label="XLM-R", scatter=False, ci = 90, line_kws={"linestyle": '-.'})
plt.legend(fontsize=9.6, prop={'size': 7.9}) # , loc='center left', bbox_to_anchor=(1, 0.5)
plt.show()

