import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from plot_tools import heatmap, annotate_heatmap
from matplotlib import colors

pairs = "VB_RBR_2_RBR_VB PPRS_NN_2_PPR_NN IN_VBG_2_IN_VBP NNS_VBP_2_NN_VBP NN_VBZ_2_NN_VBP DT_NN_2_NN_DT JJ_NN_2_NN_JJ NN_IN_2_IN_NN PPR_VBP_2_PPRS_VBP".split()


remap_pos_lbls = {"PPR":"PRP", "PPRS":"PRP$"}


pairs_fixed= []
isort = np.array([7, 4, 0, 3, 1, 2, 6, 8, 5])[::-1]
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
ds = []
for i in range(len(pairs)):
    idx = isort[i]
    try:
        f = open("../rsc/partialInhibition_lesioned/{}_{}.pkl".format(pairs[idx], corpus), "rb")
        bgs = pickle.load(f)
        f.close()


        pos_lbl = pairs[idx]
        [corr_lbl, incorr_lbl] = pos_lbl.split('_2_')
        [p1,p2] = corr_lbl.split('_')
        if p1 in remap_pos_lbls:
            p1 = remap_pos_lbls[p1]
        if p2 in remap_pos_lbls:
            p2 = remap_pos_lbls[p2]
        corr_lbl = p1 + "-" + p2
        [p1,p2] = incorr_lbl.split('_')

        if p1 in remap_pos_lbls:
            p1 = remap_pos_lbls[p1]
        if p2 in remap_pos_lbls:
            p2 = remap_pos_lbls[p2]
        incorr_lbl = p1 + "-" + p2
        pos_lbl = corr_lbl + " vs " + incorr_lbl

        pairs_fixed.append(pos_lbl)

        correct = np.array(bgs["correct"]["vlens"])[:, 0]
        incorrect = np.array(bgs["incorrect"]["vlens"])[:, 0]
        #[bg_corr, bg_icorr] = pairs[idx].split('_2_')
        g_types.append(corr_lbl)
        ug_types.append(incorr_lbl)
        grammatical[corr_lbl] = correct
        ungrammatical[incorr_lbl] = incorrect
        d = correct - incorrect
        bad_idx = np.argmin(d)
        good_idx=np.argmax(d)
        bad_examples[pairs[idx]] = (bgs["correct"]['probe'][bad_idx], bgs["incorrect"]['probe'][bad_idx], d[bad_idx])
        good_examples[pairs[idx]] = (bgs["correct"]['probe'][good_idx], bgs["incorrect"]['probe'][good_idx], d[good_idx])
        scores[pos_lbl] = d
        probes[pos_lbl] = list(zip( bgs["correct"]['probe'], bgs["incorrect"]['probe']))
        discrims.append(np.mean(d)/np.std(d))


        print(pos_lbl, len(d),np.mean(d)/np.std(d), (correct - incorrect).mean(), (correct - incorrect).std(), correct.mean(), incorrect.mean())
        ds.append(np.mean(d)/np.std(d))
    except:
        continue




#plotDists = True
#if plotDists:
#
#    dists = np.array([list(scores[pairs_fixed[i]]) + (80 - len(scores[pairs_fixed[i]]))*[0] for i in range(len(pairs_fixed))]) #padd missing with zeros
#    df = pd.DataFrame(dists.T, columns=list(map(lambda x : '-'.join(x.replace('2', 'to').split('_')) , pairs_fixed))  )
#    sns.set_palette("Purples_r")
#    subplots = df.hist()
#    for i in range(len(subplots)):
#        for j in range(len(subplots[i])):
#            subplots[i][j].axvline(0, color='red')
#            subplots[i][j].grid(False)
#            subplots[i][j].axes.get_yaxis().set_visible(False)
#            subplots[i][j].axes.get_xaxis().set_visible(False)


fig, axarr = plt.subplots(nrows =3,ncols=3)
for i in range(3):
    for j in range(3):
        pair = pairs_fixed[i*3 + j]
        axarr[i,j].hist(scores[pair], alpha = 0.5, label=pair)
        axarr[i,j].set_title(pair)
        axarr[i,j].axvline(0, color='red')
        axarr[i,j].axes.get_yaxis().set_visible(False)
        axarr[i,j].axes.get_xaxis().set_visible(False)
fig.show()
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
P = np.zeros((9,9))
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
        P[k, l] = sum(np.array(diffs) > 0)/len(diffs)
        distDiff += diffs
        l += 1
    k += 1
        
D = muDiff / (stdDiff + 1e-32)
fig = plt.figure()
(im, cbar) = heatmap(D, list(map(lambda x : '-'.join(x.replace('2', 'to').split('_')) , g_types)), 
                        list(map(lambda x : '-'.join(x.replace('2', 'to').split('_')) , ug_types)), cmap = "PRGn", cbarlabel="Discriminability")
annotate_heatmap(im, D)
fig.show()





pCorrect = {}
for i in range(len(pairs_fixed)):
    p = sum(scores[pairs_fixed[i]] > 0)/len(scores[pairs_fixed[i]])
    pCorrect[pairs_fixed[i]] = p
    print(pairs_fixed[i], p)


f = open("stimFreqs.pkl", "rb")
stimFreqs = pickle.load(f)
f.close()

frqInfo= np.zeros((9,6))
frqInfoFull = np.zeros((9,6, 80))
print("pairs", "g_fr", "g1_fr", "g2_fr", "ug_fr", "ug1_fr", "ug2_fr")
for i in range(len(pairs_fixed)):
    g, ug = stimFreqs[pairs[isort[i]]]
    
    g = np.array(g)
    ug = np.array(ug)

    [g_fr, g1_fr, g2_fr] = g.mean(axis=0)
    [ug_fr, ug1_fr, ug2_fr] = ug.mean(axis=0)
    frqInfoFull[i, :3, :] = g.T
    frqInfoFull[i, 3:, :] = ug.T

    print(pairs_fixed[i], g_fr, g1_fr, g2_fr, ug_fr, ug1_fr, ug2_fr,np.mean(scores[pairs_fixed[i]])/np.std(scores[pairs_fixed[i]]) )
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
#
f = open("discVsRT_median.txt", "r")
discVsRT = f.read().split('\n')[:-1]
f.close()

sns.set(font_scale=1.5)
labels = []
ds     = []
rts    = []
for i in range(len(discVsRT)):
    [lbl, d, rt] = discVsRT[i].split('\t')
    labels.append('-'.join(lbl.replace('2', 'to').split('_')))
    ds.append(float(d))
    rts.append(float(rt))

df = pd.DataFrame({"Comparison": labels, "Discriminability": ds, "Median RT": rts})
sns.scatterplot(data=df, x="Discriminability", y = "Median RT")
#fig, ax = plt.subplots()
#df.plot('Discriminability', 'Mean RT', kind='scatter')
for k, v in df.iterrows():
    x = v["Discriminability"]
    y = v["Median RT"]
    txt = '-'.join(v["Comparison"].split())
    plt.text(x,y, txt)


