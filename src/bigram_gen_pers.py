import sys, os
from ANet import *
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

class OBVGen(object):
    def __init__(self, k):
        '''constructs 2**k orthogonal bi-polar vectors'''
        self.N = 2**k
        v = np.array([1])
        self.E = []
        self.expand(v)

    def expand(self, v):
        v1 = np.hstack([v,v])
        v2 = np.hstack([v,-v])
        if len(v1) < self.N:
            self.expand(v1)
            self.expand(v2)
        else:
            self.E.append(v1)
            self.E.append(v2)


if __name__ == "__main__":




    N = 2**14#1024 #16360#2**16

    K = 2

    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":1e-7, 
               "eta":1,
               "alpha":1.001,
               "beta":1,
               "V0":N,
               "N":N,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"DENSat",
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

    memory_path = "TASA"#sys.argv[1]

#    sweep_idx_i = int(sys.argv[2])
#    sweep_idx_j = int(sys.argv[3])



    NSamples = 20

    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"

    f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}
    ANet.V = len(ANet.vocab)

    comp_idx = int(sys.argv[1])







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
    ANet.A = None

    ANet.COUNTS = csr_matrix(C)
    del C
    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()

    ANet.prune(min_wf = 50) #50 works



    if hparams["distributed"]:
        print("Constructing e-vecs")
        obv = OBVGen(14)
        #ANet.E = list(np.load(root_mem_path + "/{}/E{}.npy".format(memory_path, N)))

        ANet.E = obv.E[:ANet.V]#/np.linalg.norm(obv.E[:ANet.V])

    ANet.nullvec = np.zeros(ANet.V*ANet.K)




    toLoad = False
    if toLoad:
        print("Loading weight matrix")
        ANet.W = np.load(root_mem_path + "/{}/pmi.npy".format(memory_path))
        print("Loading eigenspectrum")
        ANet.ei = np.load(root_mem_path + "/{}/ei_pmi.npy".format(memory_path))
        ANet.ev = np.load(root_mem_path + "/{}/ev_pmi.npy".format(memory_path))
        ANet.W /= ANet.ei[0]
    else:
        print("Crunching out the weights...")
        ANet.compute_weights(binaryMat=False)
        #print("Saving weight matrix")
        #np.save(root_mem_path + "/{}/pmi".format(memory_path), ANet.W )
#        ANet.update_eig()
        #print("Saving eigenspectrum")
        #np.save(root_mem_path + "/{}/ei_pmi".format(memory_path), ANet.ei )
        #np.save(root_mem_path + "/{}/ev_pmi".format(memory_path), ANet.ev )



    #pbar = ProgressBar(maxval=10).start()
    #W_new = np.zeros((2*ANet.E[0].shape[0],2*ANet.E[0].shape[0]))
    #for i in range(10):        
    #    for j in range(10):
    #        x_in = np.hstack([ANet.E[i], ANet.E[j]])
    #        W_new += ANet.W[i, j]*np.outer(x_in, x_in)
    
#    pbar.update(i+1)

    #N = 1*ANet.N
    #ANet.N = ANet.V
    #ANet.W = np.array(ANet.COUNTS.todense()).astype(float)
    ANet.norm_eig(verbos=True, eps=1e-5)
    ANet.W /= (ANet.ei - 1)
    ev = ANet.ev[:, 0]
    eta = 0.55
    for i in range(ANet.V*ANet.K):
        ANet.W[i, :] -= eta*ev[i]*ev

    ANet.N = ANet.V
    
    #ANet.N = 1*N
    #E = np.array(ANet.E).T

    #W_new = np.zeros((ANet.N*ANet.K, ANet.N*ANet.K))
    #for p in range(ANet.K):
    #    for q in range(p, ANet.K):
    #        W_new[p*ANet.N:(p+1)*ANet.N, q*ANet.N:(q+1)*ANet.N] = E.dot(ANet.W[p*ANet.V:(p+1)*ANet.V, q*ANet.V:(q+1)*ANet.V]).dot(E.T) 

    #ANet.W = W_new
    ##ANet.update_eig()
    #ANet.theta = 0.01


    
    
    
    ANet.theta =1e-6
    
    








    
    toLesion = True

    pairs = "VB_RBR_2_RBR_VB PPRS_NN_2_PPR_NN IN_VBG_2_IN_VBP NNS_VBP_2_NN_VBP NN_VBZ_2_NN_VBP DT_NN_2_NN_DT JJ_NN_2_NN_JJ NN_IN_2_IN_NN PPR_VBP_2_PPRS_VBP".split()#[1:]

    runSet = sys.argv[2] == "bgs"

    if runSet:

        grammatical = "VB_RBR PPRS_NN IN_VBG NNS_VBP NN_VBZ DT_NN JJ_NN NN_IN PPR_VBP".split()
        ungrammatical="RBR_VB PPR_NN IN_VBP NN_VBP NN_VBP NN_DT NN_JJ IN_NN PPRS_VBP".split()



        pair_set = pairs[comp_idx]

        print (pair_set)


        #f = open("../rsc/bigrams/" +grammatical[i] + ".txt", "r")
        #probes_g = f.readlines()
        #f.close()

        #f = open("../rsc/bigrams/" +ungrammatical[i] + ".txt", "r")
        #probes_ug = f.readlines()
        #f.close()

        f = open("../rsc/to_run_NOVELS"+ pair_set + ".txt", "r")
        bgs = f.readlines()
        f.close()

        probes_g  = []
        probes_ug = []
        for l in range(len(bgs)):
             [corrA, corrB, incorrA, incorrB] = bgs[l].split()
             probes_g.append(corrA + " " + corrB)
             probes_ug.append(incorrA + " " + incorrB)




        scores = {"correct":{}, "incorrect":{}}
        corr_lens = []
        incorr_lens=[]

        corr_lens1 = [] ##save the first length
        incorr_lens1 = []



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

            if w1_c in ANet.I and w2_c in ANet.I and w1_i in ANet.I and w2_i in ANet.I:
                if toLesion:
                    ANet - (w1_c, w2_c)

                for i in range(len(probes)):
                    probe = probes[i]

                    ANet.probe(probe)

                    terminal = ' '.join([ANet.banks[bank_lbls[j]][0][0] for j in range(len(bank_lbls))])
                    #change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
                    cycles = ANet.count
                    if i == 0:
                        corr_lens.append(ANet.vlens[-1])
                        corr_lens1.append(ANet.vlens[1])
                    else:
                        incorr_lens.append(ANet.vlens[-1])
                        incorr_lens1.append(ANet.vlens[1])

                    if "vlens" not in scores[labels[i]]:
                        scores[labels[i]]["vlens"] = [ANet.vlens[-1]]
                        scores[labels[i]]["vlens1"] = [ANet.vlens[1]]
                        #scores[labels[i]]["ncycles"] = [len(ANet.frames)]
                        scores[labels[i]]["probe"] = [probe]
                        scores[labels[i]]["freq"] = [frq]
                    else:
                        scores[labels[i]]["vlens"].append(ANet.vlens[-1])
                        scores[labels[i]]["vlens1"].append(ANet.vlens[1])
                        #scores[labels[i]]["ncycles"].append(len(ANet.frames))
                        scores[labels[i]]["probe"].append(probe)
                        scores[labels[i]]["freq"].append(frq)

                if toLesion:
                    ~ ANet #reset


        if toLesion:
            print( "Lesioned")
        else:
            print("Not lesioned")

        corr_lens = np.array(corr_lens)
        incorr_lens = np.array(incorr_lens)
        #print("Is symmetric: ", np.sum(np.abs((ANet.W - ANet.W.T).data)) == 0)
        #ANet.print_eigenspectrum()
        print("meanCorr meanIncorr stdCorr stdIncorr meanDiff stdDiff")
        print(np.mean(corr_lens), np.mean(incorr_lens), np.std(corr_lens), np.std(incorr_lens), (corr_lens - incorr_lens).mean(), (corr_lens - incorr_lens).std(), (corr_lens - incorr_lens).mean()/(corr_lens - incorr_lens).std())

        f = open(root_mem_path + "/"+pair_set + "intact_{}_densat.pkl".format(memory_path), "wb")
        pickle.dump(scores, f)
        f.close()















































































































































































































































































































































































































































































