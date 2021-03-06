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
               "feedback":"stp",
               "init_weights":False,
               "gpu":False,
               "localist":True,
               "distributed":False,
               "maxiter":100000,
               "explicit":True}
    ANet = AssociativeNet(hparams)

    memory_path ="MIX" #sys.argv[1]

    NSamples = 100

    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"

    f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}

    f = open("log.out", "w")
    f.write("Let's go..." +"\n")
    f.close()
    totalChunks=1

    ###merging chunks
    nchunk = 1
    print ("merging chunks")
    V = len(vocab)
    C = load_csr(root_mem_path + "/{}/C_{}_{}".format(memory_path, 0, totalChunks),  (V*K, V*K), dtype=np.int64) #pre-computed
    for IDX in range(1, nchunk):
        C[:, :] += load_csr(root_mem_path + "/{}/C_{}_{}".format(memory_path, IDX, totalChunks), (V*K, V*K), dtype=np.int64)

    ANet.COUNTS = csr_matrix(C)
    del C
    print("Crunching out the weights...")
    ANet.compute_weights(binaryMat=False)
    ANet.nullvec = np.zeros((K*V))
    ANet.N = V

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()
    
    toLesion = True

    pairs = "VB_RBR_2_RBR_VB PPRS_NN_2_PPR_NN IN_VBG_2_IN_VBP NNS_VBP_2_NN_VBP NN_VBZ_2_NN_VBP DT_NN_2_NN_DT JJ_NN_2_NN_JJ NN_IN_2_IN_NN PPR_VBP_2_PPRS_VBP".split()#[1:]


    if True: 

        grammatical = "VB_RBR PPRS_NN IN_VBG NNS_VBP NN_VBZ DT_NN JJ_NN NN_IN PPR_VBP".split()
        ungrammatical="RBR_VB PPR_NN IN_VBP NN_VBP NN_VBP NN_DT NN_JJ IN_NN PPRS_VBP".split()


        for i in range(len(pairs)):
            pair_set = pairs[i]

            print (pair_set)
             

            #f = open("../rsc/bigrams/" +grammatical[i] + ".txt", "r")
            #probes_g = f.readlines()
            #f.close()

            #f = open("../rsc/bigrams/" +ungrammatical[i] + ".txt", "r")
            #probes_ug = f.readlines()
            #f.close()

            f = open("../rsc/to_run_NOVELS"+ pairs[i] + ".txt", "r")
            bgs = f.readlines()
            f.close()

            probes_g  = [] 
            probes_ug = []
            for l in range(len(bgs)):
                 [corrA, corrB, incorrA, incorrB] = bgs[l].split()
                 probes_g.append(corrA + " " + corrB)
                 probes_ug.append(incorrA + " " + incorrB)




            scores = {"correct":{}, "incorrect":{}}



            for k in range(len(bgs)):
                #(frq, [correct, incorrect]) = pair_items[k]
                frq = 0
                correct = probes_g[k]
                incorrect = probes_ug[k]


                
                print (frq, correct, incorrect)
            
                bank_lbls = ['t-0', 't-1']
              
            
                labels = ["correct", "incorrect"]
                probes = [correct, incorrect]
            
                [w1_c, w2_c] = correct.split()
                [w1_i, w2_i] = incorrect.split()
            
            
                
                if toLesion:
                    #lesion
                    if ANet.hparams['explicit']:
                        a2b = deepcopy(ANet.W[ANet.I[w1_c],ANet.N + ANet.I[w2_c]])
                        b2a = deepcopy(ANet.W[ANet.N + ANet.I[w2_c],ANet.I[w1_c]])
                        ANet.W[ANet.I[w1_c],ANet.N + ANet.I[w2_c]] = 0
                        ANet.W[ANet.N + ANet.I[w2_c],ANet.I[w1_c]] = 0
                        assert(ANet.W[ANet.I[w1_c],ANet.N + ANet.I[w2_c]] == 0)
                        assert(ANet.W[ANet.N + ANet.I[w2_c],ANet.I[w1_c]] == 0)
                    else:
                        a2b = deepcopy(ANet.WEIGHTS[0][1][ANet.I[w1_c],ANet.I[w2_c]]) ###lesion both ways
                        b2a = deepcopy(ANet.WEIGHTS[1][0][ANet.I[w2_c],ANet.I[w1_c]])
                        ANet.WEIGHTS[0][1][ANet.I[w1_c],ANet.I[w2_c]] = 0
                        ANet.WEIGHTS[1][0][ANet.I[w2_c],ANet.I[w1_c]] = 0
            
                        assert(ANet.WEIGHTS[0][1][ANet.I[w1_c], ANet.I[w2_c]] == 0) 
                        assert(ANet.WEIGHTS[1][0][ANet.I[w2_c], ANet.I[w1_c]] == 0)
            
                for i in range(len(probes)):
                    probe = probes[i]
            
                    ANet.probe(probe)
            
                    terminal = ' '.join([ANet.banks[bank_lbls[j]][0][0] for j in range(len(bank_lbls))])
                    change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
                    cycles = len(ANet.frames)
            
                    if "vlens" not in scores[labels[i]]:
                        scores[labels[i]]["vlens"] = [ANet.vlens[-1]]
                        scores[labels[i]]["ncycles"] = [len(ANet.frames)]
                        scores[labels[i]]["probe"] = [probe]
                        scores[labels[i]]["freq"] = [frq]
                    else:
                        scores[labels[i]]["vlens"].append(ANet.vlens[-1])
                        scores[labels[i]]["ncycles"].append(len(ANet.frames))
                        scores[labels[i]]["probe"].append(probe)                
                        scores[labels[i]]["freq"].append(frq)

            
            
                if toLesion: 
                    #reset
                    if ANet.hparams['explicit']:
                        ANet.W[ANet.I[w1_c],ANet.N + ANet.I[w2_c]] = deepcopy(a2b)
                        ANet.W[ANet.N + ANet.I[w2_c],ANet.I[w1_c]] = deepcopy(b2a)
                    else:
                        ANet.WEIGHTS[0][1][ANet.I[w1_c],ANet.I[w2_c]] = deepcopy(a2b)
                        ANet.WEIGHTS[1][0][ANet.I[w2_c],ANet.I[w1_c]] = deepcopy(b2a)

            if toLesion:
                print( "Lesioned")
            else:
                print("Not lesioned")

            f = open(root_mem_path + "/"+pair_set + "_MIX.pkl", "wb")
            pickle.dump(scores, f)
            f.close()




















































































































































































































































































































































































