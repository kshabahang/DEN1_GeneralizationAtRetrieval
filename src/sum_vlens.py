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
scores= {}
probes = {}
grammatical = {}
ungrammatical={}
discrims = []
ug_types = []
g_types = []
bad_examples = {}
good_examples={}
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
        discrims.append(np.mean(d)/np.std(d))
        print(pairs[i], len(d),np.mean(d)/np.std(d), (correct - incorrect).mean(), (correct - incorrect).std(), correct.mean(), incorrect.mean())
    except:
        continue

plotDists = True
if plotDists:
    dists = np.array([list(scores[pairs[i]]) + (80 - len(scores[pairs[i]]))*[0] for i in range(len(pairs))]) #padd missing with zeros
    df = pd.DataFrame(dists.T, columns=pairs)
    sns.set_palette("mako")
    subplots = df.hist()
    for i in range(len(subplots)):
        for j in range(len(subplots[i])):
            subplots[i][j].axvline(0, color='red')
            subplots[i][j].grid(False)

#fig, axarr = plt.subplots(nrows =3,ncols=3)
#for i in range(3):
#    for j in range(3):
#        pair = pairs[i*3 + j]
#        axarr[i,j].hist(scores[pair], alpha = 0.5, label=pair)
#        axarr[i,j].legend()
#        axarr[i,j].axvline(0)
#fig.show()
#
#
#f = open("bad_examples_{}.pkl".format(corpus), "wb")
#pickle.dump(bad_examples, f)
#f.close()
#
#f = open("good_examples_{}.pkl".format(corpus), "wb")
#pickle.dump(good_examples, f)
#f.close()

distDiff = []
muDiff = np.zeros((9,9))
stdDiff = np.zeros((9,9))
k = 0
for bg_g in g_types:
    l = 0
    for bg_ug in ug_types:
        fam_g  = grammatical[bg_g] 
        fam_ug = ungrammatical[bg_ug]
        diffs = []
        for i in range(len(fam_g)):
            for j in range(len(fam_ug)):
                diffs.append(fam_g[i] - fam_ug[j])
        muDiff[k, l] = np.mean(diffs)
        stdDiff[k, l] = np.std(diffs)
        distDiff += diffs
        l += 1
    k += 1
        
D = muDiff / (stdDiff + 1e-32) 

pCorrect = {}
for i in range(len(pairs)):
    p = sum(scores[pairs[i]] > 0)/len(scores[pairs[i]])
    pCorrect[pairs[i]] = p
    print(pairs[i], p)


f = open("stimFreqs.pkl", "rb")
stimFreqs = pickle.load(f)
f.close()

frqInfo= np.zeros((9,6))
frqInfoFull = np.zeros((9,6, 80))
for i in range(len(pairs)):
    g, ug = stimFreqs[pairs[i]]
    
    g = np.array(g)
    ug = np.array(ug)

    [g_fr, g1_fr, g2_fr] = g.mean(axis=0)
    [ug_fr, ug1_fr, ug2_fr] = ug.mean(axis=0)
    frqInfoFull[i, :3, :] = g.T
    frqInfoFull[i, 3:, :] = ug.T

    print(pairs[i], g_fr, g1_fr, g2_fr, ug_fr, ug1_fr, ug2_fr,np.mean(scores[pairs[i]])/np.std(scores[pairs[i]]) )
    frqInfo[i, :] = np.array([g_fr, g1_fr, g2_fr, ug_fr, ug1_fr, ug2_fr])


discrims = np.array(discrims)
frqLbls = ['grammatical', 'grammatical1', 'grammatical2', 'ungrammatical', 'ungrammatical1', 'ungrammatical2']
for i in range(len(frqLbls)):
    print(frqLbls[i], np.corrcoef(discrims, frqInfo[:, i])[0][1])


#fig = plt.figure()
#for i in range(len(pairs)):
#    plt.hist(frqInfoFull[i, 3, :], alpha = 0.5, label = pairs[i])
#
#plt.legend()
#fig.show()


#f = open("eigvals.txt", "r")
#eigvals = f.read().split('\n')
#f.close()
#
#eigvals = np.array(eigvals[:-2]).astype(float)
#df = pd.DataFrame(np.vstack([range(1,len(eigvals)+1), eigvals]).T, columns=["Rank", "Eigenvalue"])
#df["Rank"] = df["Rank"].astype(int)
#subplot = sns.barplot(x = "Rank", y = "Eigenvalue", data=df, palette='viridis')

f = open("discVsRT.txt", "r")
discVsRT = f.read().split('\n')[:-1]
f.close()

labels = []
ds     = []
rts    = []
for i in range(len(discVsRT)):
    [lbl, d, rt] = discVsRT[i].split('\t')
    labels.append(lbl)
    ds.append(float(d))
    rts.append(float(rt))

df = pd.DataFrame({"Comparison": labels, "Discriminability": ds, "Mean RT": rts})

fig, ax = plt.subplots()
df.plot('Discriminability', 'Mean RT', kind='scatter')
for k, v in df.iterrows():
    x = v["Discriminability"]
    y = v["Mean RT"]
    txt = '-'.join(v["Comparison"].split())
    plt.text(x,y, txt)


