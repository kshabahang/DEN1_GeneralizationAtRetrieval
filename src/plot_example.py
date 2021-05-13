from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


youknow = np.load("../rsc/examples/youknow.npy")
yourknow=np.load("../rsc/examples/yourknow.npy")
youknow_vlens = np.load("../rsc/examples/youknow_vlens.npy")
yourknow_vlens=np.load("../rsc/examples/yourknow_vlens.npy")
vocab = np.load("../rsc/examples/vocab_pruned.npy")
V = len(vocab)



g_sts1, g_ws1 = zip(*sorted(zip(youknow[-1, :V], vocab))[::-1][:20])
g_sts2, g_ws2 = zip(*sorted(zip(youknow[-1, V:], vocab))[::-1][:20])

weights_g = []
for i in range(len(g_ws1)):
    for j in range(len(g_ws2)):
        weights_g.append((g_ws1[i], g_ws2[j]))




ug_sts1, ug_ws1 =  zip(*sorted(zip(yourknow[-1, :V], vocab))[::-1][:20])
ug_sts2, ug_ws2 =  zip(*sorted(zip(yourknow[-1, V:], vocab))[::-1][:20])

weights_ug = []
for i in range(len(ug_ws1)):
    for j in range(len(ug_ws2)):
        weights_ug.append((ug_ws1[i], ug_ws2[j]))
#topK = 10
#activation = []
#slot       = []
#word       = []
#iteration  = []
#frames = 1*youknow
#(n,m) = youknow.shape
#for i in range(n):
#    #slot 1
#    top = list(filter(lambda s : s[0] > 0, sorted(zip(frames[i, :V],vocab))[::-1][1:topK]))
#    if len(top) > 0:
#        top_strengths, top_words = zip(*top)
#        iteration += [i]*len(top)
#        slot += [1]*len(top)
#        activation += list(top_strengths)
#        word += list(top_words)
#
#    #slot 2
#    top = list(filter(lambda s : s[0] > 0, sorted(zip(frames[i, V:],vocab))[::-1][1:topK]))
#    if len(top) > 0:
#        top_strengths, top_words = zip(*top)
#        iteration += [i]*len(top)
#        slot += [2]*len(top)
#        activation += list(top_strengths)
#        word += list(top_words)


#df = pd.DataFrame({"Activation": activation, "Word":word, "Slot": slot, "Iteration": iteration})
#
#
#sns.set_theme(style="ticks")
#palette = sns.color_palette("Set2")
#plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", palette="dark", linewidth=5)
#
#top1 = list(filter(lambda s : s[0] > 0, sorted(zip(frames[-1, :V],vocab))[::-1][1:topK]))
#top2 = list(filter(lambda s : s[0] > 0, sorted(zip(frames[-1, V:],vocab))[::-1][1:topK]))
#
#for i in range(len(top1)):
#    plot.axes[0][0].text(len(frames), top1[i][0], top1[i][1], size=15)
#plot.axes[0][0].axhline(0.0003, color = 'red')
#
#for i in range(len(top2)):
#    plot.axes[0][1].text(len(frames), top2[i][0], top2[i][1], size=15)
#plot.axes[0][1].axhline(0.0003, color = 'red')
#
#
#plt.show()
##plot.set(ylim=(0,1))
##plot.set(xlim=(0, min(len(frames_new), len(frames_odd))))
##plot.set_xticklabels(range(min(len(frames_new), len(frames_odd))))
#
#activation = []
#slot       = []
#word       = []
#iteration  = []
#frames = 1*yourknow
#(n,m) = yourknow.shape
#for i in range(n):
#    #slot 1
#    top = list(filter(lambda s : s[0] > 0, sorted(zip(frames[i, :V],vocab))[::-1][1:topK]))
#    if len(top) > 0:
#        top_strengths, top_words = zip(*top)
#        iteration += [i]*len(top)
#        slot += [1]*len(top)
#        activation += list(top_strengths)
#        word += list(top_words)
#
#    #slot 2
#    top = list(filter(lambda s : s[0] > 0, sorted(zip(frames[i, V:],vocab))[::-1][1:topK]))
#    if len(top) > 0:
#        top_strengths, top_words = zip(*top)
#        iteration += [i]*len(top)
#        slot += [2]*len(top)
#        activation += list(top_strengths)
#        word += list(top_words)
#
#
#df = pd.DataFrame({"Activation": activation, "Word":word, "Slot": slot, "Iteration": iteration})
#
#
#sns.set_theme(style="ticks")
#palette = sns.color_palette("Set2")
#plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", palette="dark", linewidth=5)
#
#top1 = list(filter(lambda s : s[0] > 0, sorted(zip(frames[-1, :V],vocab))[::-1][1:topK]))
#top2 = list(filter(lambda s : s[0] > 0, sorted(zip(frames[-1, V:],vocab))[::-1][1:topK]))
#
#for i in range(len(top1)):
#    plot.axes[0][0].text(len(frames), top1[i][0], top1[i][1], size=15)
#plot.axes[0][0].axhline(0.0003, color = 'red')
#
#for i in range(len(top2)):
#    plot.axes[0][1].text(len(frames), top2[i][0], top2[i][1], size=15)
#plot.axes[0][1].axhline(0.0003, color = 'red')
#
#plt.show()
#
#

sns.set(font_scale=1.5)
sns.set_palette("viridis_r")
K = 11
vlens = np.hstack([youknow_vlens[:K, 0], yourknow_vlens[:K, 0]])
iteration = list(range(K)) + list(range(K))
probe = ["you know"]*K + ["your know"]*K
df = pd.DataFrame({"Familiarity": vlens, "Probe":probe, "Iteration":iteration})

plot = sns.relplot(data = df, x='Iteration', y='Familiarity', hue= "Probe", style="Probe", kind="line", linewidth=5)

plot.set(xticks=range(0,K))

plt.show()










