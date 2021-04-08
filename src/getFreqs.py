import sys, os
from AssociativeNet import *
#from matplotlib import pyplot as plt
#plt.ion()
from progressbar import ProgressBar

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

    N =2**17#16360#2**16

    K = 2#2

    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":1e-7, 
               "eta":0.55,
               "alpha":1.001,
               "beta":1,
               "V0":N,
               "N":N,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"stp",
               "init_weights":False,
               "gpu":False,
               "localist":True,
               "distributed":False,
               "maxiter":1000,
               "explicit":True,
               "sparse":False,
               "row_center":False,
               "col_center":False,
               "norm":"pmi"}
    ANet = AssociativeNet(hparams)
    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"
    memory_path ="TASA" #sys.argv[1]
    f = open(root_mem_path + "/" + memory_path + "/bigram_freqs.pkl", "rb")
    bgFreqs = pickle.load(f)
    f.close()
   
    f = open(root_mem_path + "/" + memory_path + "/vocab.txt", "r")
    vocab = f.read().split('\n')
    f.close()

    f = open(root_mem_path + "/" + memory_path + "/unigram_counts.pkl", "rb")
    unigram_counts = pickle.load(f)
    f.close()

    unigram_counts = {vocab[i]:unigram_counts[i] for i in range(len(vocab))}







    pairs = "VB_RBR_2_RBR_VB PPRS_NN_2_PPR_NN IN_VBG_2_IN_VBP NNS_VBP_2_NN_VBP NN_VBZ_2_NN_VBP DT_NN_2_NN_DT JJ_NN_2_NN_JJ NN_IN_2_IN_NN PPR_VBP_2_PPRS_VBP".split()#[1:]

    frqByType = {}


    for i in range(len(pairs)):
        pair_set = pairs[i]

        f = open("../rsc/to_run_NOVELS"+ pair_set + ".txt", "r")
        bgs = f.readlines()
        f.close()

        probes_g  = [] 
        probes_ug = []
        for l in range(len(bgs)):
             [corrA, corrB, incorrA, incorrB] = bgs[l].split()
             probes_g.append(corrA + " " + corrB)
             probes_ug.append(incorrA + " " + incorrB)

        g_freqs = []
        ug_freqs= []

        for k in range(len(bgs)):
            frq = 0
            g = probes_g[k]
            ug = probes_ug[k]

            if g in bgFreqs:
                g_frq = bgFreqs[g]
            else:
                g_frq = 0
            if ug in bgFreqs:
                ug_frq = bgFreqs[ug]
            else:
                ug_frq = 0
            [g1, g2] = g.split()
            [ug1,ug2] = ug.split()
            if g1 in unigram_counts:
                g1_frq = unigram_counts[g1]
            else:
                g1_frq = 0
            if g2 in unigram_counts:
                g2_frq = unigram_counts[g2]
            else:
                g2_frq = 0
            if ug1 in unigram_counts:
                ug1_frq = unigram_counts[ug1]
            else:
                ug1_frq = 0
            if ug2 in unigram_counts:
                ug2_frq = unigram_counts[ug2]
            else:
                ug2_frq = 0

            print(g, g1, g2, ug, ug1, ug2)
            print(g_frq, g1_frq, g2_frq, ug_frq, ug1_frq, ug2_frq)
            g_freqs.append([g_frq, g1_frq,g2_frq])
            ug_freqs.append([ug_frq, ug1_frq, ug2_frq])
        frqByType[pair_set] = [g_freqs, ug_freqs]

    f = open("stimFreqs.pkl", "wb")
    pickle.dump(frqByType, f)
    f.close()














































































































































































































































































































































































































































