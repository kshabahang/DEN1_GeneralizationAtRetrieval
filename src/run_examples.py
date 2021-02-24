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
               "eps":1e-10, 
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
               "maxiter":1000,
               "explicit":True,
               "multiDegree":False}
    ANet = AssociativeNet(hparams)

    memory_path ="TASA" #sys.argv[1]

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


    #maxds =[5,10,15,20,50,100,200,300,500,1000]
    maxds = [5]#[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    Ds = []
    for i in range(len(maxds)):
        MAXD = maxds[i]
        D = load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path, MAXD,0, totalChunks),  (V*K, V*K), dtype=np.int64) #pre-computed
        for IDX in range(1, nchunk):
            D[:, :] += load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path,MAXD, IDX, totalChunks), (V*K, V*K), dtype=np.int64)
        
        Ds.append( csr_matrix(D) )


    ANet.Ds = Ds
    ANet.D = ANet.Ds[0]

    ANet.COUNTS = csr_matrix(C)
    del C
    print("Crunching out the weights...")
    ANet.compute_weights(binaryMat=False)
    ANet.nullvec = np.zeros((K*V))
    ANet.N = V

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()

    f = open("bad_examples_TASA.pkl", "rb")
    bad_examples = pickle.load(f)
    f.close()

    for comp in bad_examples.keys():
        corr, incorr, d = bad_examples[comp]
        
        ANet.lesion(tuple(corr.split()))

        ANet.probe(corr)
        ANet.get_top_connect()
        lbl = comp +"_" + "_".join(corr.split()) + "_corr"
        ANet.save_top_weights(lbl)

        ANet.probe(incorr)
        ANet.lesion(tuple(incorr.split()))
        ANet.get_top_connect()
        lbl = comp +"_" + "_".join(incorr.split()) + "_incorr"
        ANet.save_top_weights(lbl)

        ANet.reverse_lesion()














































































































































































































































































































































































































































































































































