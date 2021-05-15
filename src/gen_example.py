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

    memory_path ="TASA" #sys.argv[1]

    NSamples = 100

    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"

    f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}
    ANet.V = len(vocab)

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


    f = open(root_mem_path + "/{}/A_log.txt".format(memory_path), "r")
    A_dims = f.read().split()
    f.close()
    A_shape = (int(A_dims[0]), int(A_dims[1]))


    
    A = load_csr(root_mem_path + "/{}/A".format(memory_path),  A_shape, dtype=np.int32) #pre-computed



    #maxds =[5,10,15,20,50,100,200,300,500,1000]
    maxds = [50]#[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    Ds = []
    for i in range(len(maxds)):
        MAXD = maxds[i]
        D = load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path, MAXD,0, totalChunks),  (V*K, V*K), dtype=np.int64) #pre-computed
        for IDX in range(1, nchunk):
            D[:, :] += load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path,MAXD, IDX, totalChunks), (V*K, V*K), dtype=np.int64)
        
        Ds.append( csr_matrix(D) )


    ANet.Ds = Ds
    ANet.D = ANet.Ds[0]
    ANet.A = A


    ###take out any counts less than a criterion
    #min_cc = 2
    #for i in range(len(C.data)):
    #    if C.data[i] < min_cc:
    #        C.data[i] = 0
    #C.eliminate_zeros()


    ANet.COUNTS = csr_matrix(C)



    del C
    #sys.exit()
    ##drop low freq term
    ANet.prune(min_wf = 50) #50 works

    toLoad = True
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
        print("Saving weight matrix")
        np.save(root_mem_path + "/{}/pmi".format(memory_path), ANet.W )
        ANet.update_eig()
        print("Saving eigenspectrum")
        np.save(root_mem_path + "/{}/ei_pmi".format(memory_path), ANet.ei )
        np.save(root_mem_path + "/{}/ev_pmi".format(memory_path), ANet.ev )



    #i_max = 40
    #for i in range(i_max):
    #    neg_w = (ANet.ei[i] - ANet.ei[i_max])/ANet.ei[i]
    #    ANet.W -= neg_w*np.outer(ANet.ev[:, i], ANet.ev[:, i])


    #ANet.map_eig2weight(k=1000)
    #np.save("Emap", ANet.EMap)
    #f = open("emap_ws.pkl", "wb")
    #pickle.dump(ANet.emap_ws, f)
    #f.close()

    #e_max = 75.4832 #change this if you change the learning rule
    #ANet.alpha = 1.001
    #ANet.W /= e_max
    ANet.nullvec = np.zeros((K*ANet.V))
    ANet.N = ANet.V

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()
    
    toLesion = True


    #target = "buffalo"

    #bg_corr = "her " + target
    #bg_incorr= "she " + target

    #target = ""
    #context=  "snake"
    #                            
    #bg_corr = "{}s ".format(context) + target
    #bg_incorr= "{} ".format(context) + target



    [corr1, corr2] = bg_corr.split()

    ANet - (corr1, corr2)

    ANet.probe(bg_corr)

    vlens_corr = np.array(ANet.vlens)[:, 0]

    g1 = ANet.banks['t-0']
    g2 = ANet.banks['t-1']

    Wg = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            Wg[i, j] = ANet.W[ANet.I[g1[0][i]], ANet.V + ANet.I[g2[0][j]]]


    ANet.probe(bg_incorr)

    vlens_incorr= np.array(ANet.vlens)[:, 0]

    ug1 = ANet.banks['t-0']
    ug2 = ANet.banks['t-1']

    Wug = np.zeros((20,20))

    Wug = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            Wug[i, j] = ANet.W[ANet.I[ug1[0][i]], ANet.V + ANet.I[ug2[0][j]]]

    ~ ANet

    k = min([len(vlens_corr), len(vlens_incorr)])

    print(vlens_corr[:k] - vlens_incorr[:k])

    np.save("g1", g1)
    np.save("g2", g2)
    np.save("ug1", ug1)
    np.save("ug2", ug2)
    np.save("Wg", Wg)
    np.save("Wug", Wug)































































































































































































































































































































































































































































































