import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from plot_tools import heatmap, annotate_heatmap
from matplotlib import colors

pairs = "VB_RBR_2_RBR_VB PPRS_NN_2_PPR_NN IN_VBG_2_IN_VBP NNS_VBP_2_NN_VBP NN_VBZ_2_NN_VBP DT_NN_2_NN_DT JJ_NN_2_NN_JJ NN_IN_2_IN_NN PPR_VBP_2_PPRS_VBP".split()

corpus = sys.argv[1]

f = open("../rsc/byItemHuman.pkl", "rb")
RTs = dict(pickle.load(f))
f.close()

scores= {}
probes = {}
grammatical = {}
ungrammatical={}
discrims = []
ug_types = []
g_types = []
bad_examples = {}
good_examples={}
rts_and_diff= []
for i in range(len(pairs)):
    try:
        f = open("../rsc/partialInhibition_intact/{}_{}.pkl".format(pairs[i], corpus), "rb")
        bgs = pickle.load(f)
        f.close()

        correct = np.array(bgs["correct"]["vlens"])[:, 0]
        incorrect = np.array(bgs["incorrect"]["vlens"])[:, 0]
        [bg_corr, bg_icorr] = pairs[i].split('_2_')
        g_types.append(bg_corr)
        ug_types.append(bg_icorr)
        grammatical[bg_corr] = correct
        ungrammatical[bg_icorr] = incorrect
        d = correct - incorrect
        bad_idx = np.argmin(d)
        good_idx=np.argmax(d)
        bad_examples[pairs[i]] = (bgs["correct"]['probe'][bad_idx], bgs["incorrect"]['probe'][bad_idx], d[bad_idx])
        good_examples[pairs[i]] = (bgs["correct"]['probe'][good_idx], bgs["incorrect"]['probe'][good_idx], d[good_idx])
        scores[pairs[i]] = d
        probes[pairs[i]] = list(zip( bgs["correct"]['probe'], bgs["incorrect"]['probe']))
        rts_i = [np.median(np.array(list(filter(lambda x : x[1] == 1, RTs[pair])))[:, 0]) for pair in probes[pairs[i]]]
        rts_and_diff.append(np.vstack([rts_i, (d - d.mean())/d.std()]))

        discrims.append(np.mean(d)/np.std(d))
        print(pairs[i], len(d),np.mean(d)/np.std(d), (correct - incorrect).mean(), (correct - incorrect).std(), correct.mean(), incorrect.mean())
    except:
        continue

