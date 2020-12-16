import sys, os
from AssociativeNet import *
from matplotlib import pyplot as plt
plt.ion()

from plot_tools import *

from progressbar import ProgressBar

from matplotlib.colors import cnames
import pandas as pd
import seaborn as sns
from pylab import rc 

import pickle
from copy import deepcopy
from collections import Counter
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

def save_csr(matrix, filename):
    matrix.data.tofile(filename + "_"+ str(type(matrix.data[0])).split(".")[-1].strip("\'>") + ".data")
    matrix.indptr.tofile(filename + "_"+ str(type(matrix.indptr[0])).split(".")[-1].strip("\'>") + ".indptr")
    matrix.indices.tofile(filename + "_" +str(type(matrix.indices[0])).split(".")[-1].strip("\'>") + ".indices")

def load_csr(fpath, shape, dtype=np.float64, itype=np.int32):
    data = np.fromfile(fpath + "_{}.data".format(str(dtype).split(".")[-1].strip("\'>")), dtype)
    indptr = np.fromfile(fpath + "_{}.indptr".format(str(itype).split(".")[-1].strip("\'>")), itype)
    indices = np.fromfile(fpath + "_{}.indices".format(str(itype).split(".")[-1].strip("\'>")), itype)

    return csr_matrix((data, indices, indptr), shape = shape)


def proj(a, v):
    return (a.dot(v)/v.dot(v))*v

if __name__ == "__main__":

    N  = 4
    V0 = N
    K = 2
    
    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":0.0000001, 
               "eta":1,
               "alpha":1,
               "beta":1,
               "V0":V0,
               "N":N,
               "localist":True,
               "distributed":False,
               "init_weights":True,
               "explicit":True,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"saturate", #linear / saturate (BSB) / stp (DEN)
               "gpu":False}
    ANet = AssociativeNet(hparams)
    
    strength = 0.5
     #feedback = "linear"#sys.argv[1]
    noise =0.1#0.2#float(sys.argv[2])
    
    p_a = "the cat"
    p_b = "a dog"
    
    ANet.nullvec = np.zeros(N*K)
    
    if hparams["distributed"]:
        wvecs = {"the":[1,1,-1,-1], "cat":[-1,1,-1,1], "a":[1,1,1,1], "dog":[1,-1,-1,1]}
        wvecs = {w:wvecs[w]/np.linalg.norm(wvecs[w]) for w in wvecs.keys()}
        ANet.update_vocab(p_a.split(), wvecs = wvecs)
        ANet.update_vocab(p_b.split(), wvecs = wvecs)
    else:
        ANet.update_vocab(p_a.split())
        ANet.update_vocab(p_b.split())
    
    ANet.encode(p_a, st=1.2)
    ANet.encode(p_b, st=1.1)
    
    ei, ev = np.linalg.eig(ANet.W)
    isort = np.argsort(ei)[::-1]
    ei = ei[isort]
    ev = ev[:, isort]
    
    probes = {"old_strong":p_a, 
              "old_weak":p_b,
              "old_strong_part":"the",
              "old_weak_part":"a",
              "new1":"the dog", 
              "new2":"a cat", 
              "odd1":"cat a",
              "odd2":"dog the"}
    
    conds = "old_strong old_weak old_strong_part old_weak_part new1 new2 odd1 odd2".split()
    # conds = "new1 odd2".split()
    #conds = "old_strong_part old_weak_part new1 new2 odd1 odd2".split()[:2]
    niter = 10
    results_by_cond = {}
    #conds = "old_strong_part old_weak_part".split()

    probs = []
    reports_cond = []
    loadings_cond = []
    echos_cond = []
    frames_by_probeANDresp = {}
    sym_map = dict(zip(["the", "a", "cat", "dog"], ["tri_up", "tri_down", "circle", "square"]))
    for i in range(len(conds)):
        print() 
        print ("*"*72)
        print (conds[i])
        probe = probes[conds[i]]
    
        s1 = [0,0,0,0]
        s2 = [0,0,0,0]
    
        probe = probes[conds[i]]
        V = len(ANet.vocab)
    
        reports = {}
        loadings = {}
        echos = {}
    
        for k in range(niter):
    
            ANet.probe(probe, st = strength, noise = noise)

            x0 = ANet.frames[0]/np.linalg.norm(ANet.frames[0])
            xn = ANet.frames[-1]/np.linalg.norm(ANet.frames[-1])
            W_prime = ANet.W + np.outer(x0, x0)
            ei, ev = np.linalg.eig(W_prime)
            lambda_inf = ei.real.max()
            print(lambda_inf," = ",lambda_inf - x0.dot(xn)**2, " + ", x0.dot(xn)**2)

    
            ###terminal state
            tstate = ANet.frames[-1]
            s1[np.argmax(tstate[:V])] += 1
            s2[np.argmax(tstate[V:])] += 1
    
            report = ANet.vocab[np.argmax(tstate[:V])] + " " + ANet.vocab[np.argmax(tstate[V:])]
    
            if report not in frames_by_probeANDresp:
                frames_by_probeANDresp[probe + " " + report] = [ANet.frames]
            else:
                frames_by_probeANDresp[probe + " " + report].append(ANet.frames)
    
            if report not in reports:
                reports[report] = 1
                loads = []
                for j in range(len(ANet.echo_frames)):
                    echo = ANet.echo_frames[j]
                    loads.append([echo.dot(ev[:, 0]), echo.dot(ev[:, 1])])
                loadings[report] = [loads]
                #loadings[report] = [[echo.dot(ev[:, 0]), echo.dot(ev[:, 1])]]
                #echos[report] = [ANet.echo_frames]
            else:
                reports[report] += 1
                loads = []
                for j in range(len(ANet.echo_frames)):
                    echo = ANet.echo_frames[j]
                    loads.append([echo.dot(ev[:, 0]), echo.dot(ev[:, 1])])
                loadings[report].append(loads)
    
                #loadings[report].append( [echo.dot(ev[:, 0]), echo.dot(ev[:, 1])] )
                #echos[report].append(ANet.echo_frames)
        loadings_cond.append(loadings)
        for report in reports.keys():
            reports[report] /= niter
        results_by_cond[conds[i]] = np.array([s1, s2])/float(niter)
        reports_cond.append(reports)
    
        print ("probing with '" + probe + "'")
        ANet.probe(probe, st = strength, noise = noise)
     
    res = list(zip(map(lambda cond : probes[cond], conds), reports_cond))
    
    headers = ["the cat", "a dog", "the a", "a a", "the dog", "a cat", "dog the", "cat a"]
    
    for i in range(len(res)):
        h = ', '.join(headers)
        s = ''
        for j in range(len(headers)):
            if headers[j] in res[i][1]:
                if j != 0:
                    s += ", " + str(res[i][1][headers[j]]) 
                else:
                    s += str(res[i][1][headers[j]])
            else:
                if j != 0:
                    s += ", 0 "
                else:
                    s += "0 "
        left = list(set(res[i][1].keys()) - set(headers))
        if len(left) > 0:
            for j in range(len(left)):
                h += ", " + left[j] 
                s += ", " + str(res[i][1][left[j]]) 
        
        print ()
        print (res[i][0] + ', '+ h)
        print (res[i][0] + ', '+s)
        print()
                    
#    rc('axes',linewidth=100)

    if True:#hparams["feedback"] == "stp":
        up_left = np.array(frames_by_probeANDresp['the dog the dog'])[0][:17, :4]
        up_right = np.array(frames_by_probeANDresp['the dog the dog'])[0][:17, 4:]
    
        activation = []
        word       = []
        slot       = []
        n          = []
        for i in range(len(up_left.T)):
            activation += list(up_left[:, i])
            word += [ANet.vocab[i]]*len(up_left)
            slot += [1]*len(up_left)
            n += [j for j in range(len(up_left))]
    
        for i in range(len(up_right.T)):
            activation += list(up_right[:, i])
            word += [ANet.vocab[i]]*len(up_right)
            slot += [2]*len(up_right)
            n += [j for j in range(len(up_right))]
    
    
        df = pd.DataFrame({"Activation": activation, "Word": word, "Slot":slot, "Iteration":n})
    
        sns.set_theme(style="ticks")
        palette = sns.color_palette("Set2")
        plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", palette="dark", linewidth=4)
        plot.set(ylim=(0,0.8))

    
        if hparams['feedback'] == 'stp':        
            bottom_left = np.array(frames_by_probeANDresp['dog the the cat'])[0][:17, :4]
            bottom_right = np.array(frames_by_probeANDresp['dog the the cat'])[0][:17, 4:]
        elif hparams['feedback'] == 'saturate':
            bottom_left = np.array(frames_by_probeANDresp['dog the dog the'])[0][:17, :4]
            bottom_right = np.array(frames_by_probeANDresp['dog the dog the'])[0][:17, 4:]

    
    
        activation = []
        word       = []
        slot       = []
        n          = []
        for i in range(len(bottom_left.T)):
            activation += list(bottom_left[:, i])
            word += [ANet.vocab[i]]*len(bottom_left)
            slot += [1]*len(bottom_left)
            n += [j for j in range(len(bottom_left))]
    
        for i in range(len(bottom_right.T)):
            activation += list(bottom_right[:, i])
            word += [ANet.vocab[i]]*len(bottom_right)
            slot += [2]*len(bottom_right)
            n += [j for j in range(len(bottom_right))]
    
    
        df = pd.DataFrame({"Activation": activation, "Word": word, "Slot":slot, "Iteration":n})
    
        sns.set_theme(style="ticks")
        palette = sns.color_palette("rocket_r")

        plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", palette="dark", linewidth=4)
        plot.set(ylim=(0,0.8))




















































































































































































































































































































































































































































































































































































































































