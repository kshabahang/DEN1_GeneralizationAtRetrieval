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

    N =200 #16360#2**16

    K = 2

    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":0.0000001, 
               "eta":1,
               "alpha":1,
               "beta":1,
               "V0":N,
               "N":N,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"saturate",
               "init_weights":False,
               "gpu":False,
               "localist":False,
               "distributed":True,
               "maxiter":100,
               "explicit":False}
    ANet = AssociativeNet(hparams)

    memory_path = sys.argv[1]

#    sweep_idx_i = int(sys.argv[2])
#    sweep_idx_j = int(sys.argv[3])



    NSamples = 150

    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"

    f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}


    f = open("/home/ubuntu/word2vec/vocab.txt", "r")
    vocab_w2v = list(map(lambda line : line.strip().split()[0], f.readlines()))
    f.close()
    I_w2v = {vocab_w2v[i]:i for i in range(len(vocab_w2v))}

    w2v = np.load("/home/ubuntu/word2vec/word2vec.npy")

    for i in range(len(ANet.vocab)):
        if ANet.vocab[i] in vocab_w2v:
            v = w2v[I_w2v[ANet.vocab[i]]]
            ANet.E.append(v/norm(v))
        else:
            ANet.E.append(np.random.normal(0, 1/N, N))

    
        

    sparsifyEVECS = False
    if sparsifyEVECS:
        E_sp = []
        for i in range(len(ANet.E)):
            E_sp.append(csr_matrix(ANet.E[i]))
        ANet.E = E_sp

    f = open("log.out", "w")
    f.write("Let's go..." +"\n")
    f.close()
    totalChunks=1

    #sweep_idx_i = 0 #index for grammatical bg
    #sweep_idx_j = 0 #index for ungrammatical bg

    ###merging chunks
    nchunk = 1
    print ("merging chunks")
    V = len(vocab)
    C = load_csr(root_mem_path + "/{}/C_{}_{}".format(memory_path, 0, totalChunks),  (V*K, V*K), dtype=np.int64) #pre-computed
    for IDX in range(1, nchunk):
        C[:, :] += load_csr(root_mem_path + "/{}/C_{}_{}".format(memory_path, IDX, totalChunks), (V*K, V*K), dtype=np.int64)

#    C -= np.diag(np.array(C.diagonal()).flatten())

    ANet.COUNTS = csr_matrix(C)
    del C
    print("Crunching out the weights...")
    ANet.compute_weights(binaryMat=True)
    ANet.nullvec = np.zeros((K*N))
#    ANet.nullvec = lil_matrix((1, K*N))

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()

    
    toLesion = True

    if True: 

        grammatical = "VB_RBR PPRS_NN IN_VBG NNS_VBP NN_VBZ DT_NN JJ_NN NN_IN PPR_VBP".split()
        ungrammatical="RBR_VB PPR_NN IN_VBP NN_VBP NN_VBP NN_DT NN_JJ IN_NN PPRS_VBP".split()

        #for idx_gram in range(len(grammatical)):

        #for idx_ugram in range(len(ungrammatical)):

        #pair_set = pairs[sweep_idx]
        #pair_set = grammatical[sweep_idx_i] + "_2_" + ungrammatical[sweep_idx_j]
        #pair_set = grammatical[idx_gram] + "_2_" + ungrammatical[idx_ugram]
        print(sys.argv[2])

        f = open("../rsc/bigrams/" +sys.argv[2] + ".txt", "r")
        probes = f.readlines()
        f.close()

        
        scores = {} # {"correct":{}, "incorrect":{}}
        
        bank_lbls = ['t-0', 't-1']

        for i in range(len(probes[:NSamples])):
        
            probe = probes[i]
        
            [w1, w2] = probe.split()
 
            if toLesion:
                #lesion
                if ANet.hparams['explicit']:
                    a2b = deepcopy(ANet.W[ANet.I[w1],ANet.N + ANet.I[w2]])
                    b2a = deepcopy(ANet.W[ANet.N + ANet.I[w2],ANet.I[w1]])
                    ANet.W[ANet.I[w1],ANet.N + ANet.I[w2]] = 0
                    ANet.W[ANet.N + ANet.I[w2],ANet.I[w1]] = 0
                    assert(ANet.W[ANet.I[w1],ANet.N + ANet.I[w2]] == 0)
                    assert(ANet.W[ANet.N + ANet.I[w2],ANet.I[w1]] == 0)
                else:
                    a2b = deepcopy(ANet.WEIGHTS[0][1][ANet.I[w1],ANet.I[w2]]) 
                    b2a = deepcopy(ANet.WEIGHTS[1][0][ANet.I[w2],ANet.I[w1]])
                    ANet.WEIGHTS[0][1][ANet.I[w1],ANet.I[w2]] = 0
                    ANet.WEIGHTS[1][0][ANet.I[w2],ANet.I[w1]] = 0
            
                    assert(ANet.WEIGHTS[0][1][ANet.I[w1], ANet.I[w2]] == 0) 
                    assert(ANet.WEIGHTS[1][0][ANet.I[w2], ANet.I[w1]] == 0)
         
            
        
            ANet.probe(probe, toNorm = True)
        
            terminal = ' '.join([ANet.banks[bank_lbls[j]][0][0] for j in range(len(bank_lbls))])
            #change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
            #cycles = len(ANet.frames)
        
            if "vlens" not in scores:
                scores["vlens"] = [ANet.vlens]
                scores["ncycles"] = [ANet.count]
              #  scores[labels[i]]["change"] = [change]
              #  scores[labels[i]]["terminal"] = [terminal]
                scores["probe"] = [probe]
#                scores["freq"] = [frq]
              #  scores[labels[i]]["frames"] = [ANet.frames]
            else:
                scores["vlens"].append(ANet.vlens)
                scores["ncycles"].append(ANet.count)
               # scores[labels[i]]["change"].append(change)
               # scores[labels[i]]["terminal"].append(terminal)
                scores["probe"].append(probe)                
#                scores["freq"].append(frq)
              #  scores[labels[i]]["frames"].append(ANet.frames)
        
        
        if toLesion: 
            #reset
            if ANet.hparams['explicit']:
                ANet.W[ANet.I[w1],ANet.N + ANet.I[w2]] = deepcopy(a2b)
                ANet.W[ANet.N + ANet.I[w2],ANet.I[w1]] = deepcopy(b2a)
            else:
                ANet.WEIGHTS[0][1][ANet.I[w1],ANet.I[w2]] = deepcopy(a2b)
                ANet.WEIGHTS[1][0][ANet.I[w2],ANet.I[w1]] = deepcopy(b2a)


        if toLesion:
            print( "Lesioned")
        else:
            print("Not lesioned")

        f = open(root_mem_path + "/"+sys.argv[2] + "_output_bsb.pkl", "wb")
        pickle.dump(scores, f)
        f.close()




















































































































































































































































































































































































