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

    K = 2

    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":0.01, 
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
               "gpu":False,
               "maxiter":100000,
                "localist":True,
                "init_weights":False,
                "distributed":False,
                "explicit":False,
                "sparse":False,
                "row_center":False,
                "col_center":False}
    ANet = AssociativeNet(hparams)

    MODE = sys.argv[1]

    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"

    #MAXDs = [5,10,15,20,50,100,200,300,500,1000]
    MAXDs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]


    if MODE == "init":

        memory_path = sys.argv[2]

        print ("Number of banks: {} ---- Feature dimensionality: {}".format(K, N))
        print ("Initializing...")

        ###load corpus
        f = open(root_mem_path + "/{}/{}.txt".format(memory_path, memory_path), "r")
        corpus = f.readlines()
        f.close()
        corpus = corpus
        corpus_clean = []
        print ("Loading corpus...")
        pbar = ProgressBar(maxval=len(corpus)).start()
        for i in range(len(corpus)):
            if corpus[i] != '\n' and len(corpus[i].split()) > 1:
                corpus_clean.append(corpus[i])
                ANet.update_vocab(corpus[i].split())
            pbar.update(i)

#        ANet.E = np.array(ANet.E)
#        for i in range(len(ANet.E)):
#            ANet.E[i] = ANet.E[i]/np.linalg.norm(ANet.E[i])
#        np.savez("E{}".format(N), ANet.E)
        f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "w")
        f.write("\n".join(ANet.vocab))
        f.close()

    if MODE == "count":
        ###compute the count matrix
        IDX = int(sys.argv[2])
        CHU = int(sys.argv[3])
        memory_path = sys.argv[4] #where to dump the Counts

        f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
        vocab = f.readlines()
        f.close()
        vocab = [vocab[i].strip() for i in range(len(vocab))]
        I = {vocab[i]:i for i in range(len(vocab))}
        V = len(vocab)

        f = open(root_mem_path + "/{}/{}.txt".format(memory_path, memory_path), "r")
        corpus = f.readlines()
        f.close()
        L = len(corpus)/CHU
        corpus = corpus[int(IDX*L):int((IDX+1)*L)]
        corpus_int = []
        print("Loading corpus...")
        pbar = ProgressBar(maxval=len(corpus)).start()
        for i in range(len(corpus)):
            line = corpus[i].strip().split()
            if len(line) > 1:
                corpus_int += [I[line[j]] for j in range(len(line)) if line[j] != "_"]
            
            pbar.update(i)


        print("Computing bi-gram counts...")
#        C = np.zeros((K*V, K*V))
        C = lil_matrix((K*V, K*V), dtype=np.int64)

        Ds = [lil_matrix((K*V, K*V), dtype=np.int64) for m in range(len(MAXDs))]

        print("Constructing data matrix")
        pbar = ProgressBar(maxval = len(corpus_int) - 1).start()
        A = lil_matrix((len(corpus_int) - 1, K*V), dtype=np.int32) #data matrix
        for i in range(len(corpus_int) - 1):
            A[i, corpus_int[i]] = A[i, V + corpus_int[i+1]] = 1
            pbar.update(i+1)

        pbar = ProgressBar(maxval=K**2).start()
        for k in range(K):
            for l in range(k,K):
                c = Counter([(corpus_int[i+k], corpus_int[i+l]) for i in range(len(corpus_int) - max(k, l) - 1)])
                for m in range(len(MAXDs)): 
                    degrees = np.zeros(V)
                    for ((i1, i2), frq) in c.most_common():
                        if i1 != i2 and degrees[i1] <= MAXDs[m] and degrees[i2] <= MAXDs[m]:
                            Ds[m][int(k*V)+i1, int(l*V) + i2] = frq
#                            D[int(l*V) + i2, int(k*V)+i1] = frq
                            degrees[i1] += 1
                            degrees[i2] += 1

                idx = c.keys()
#                C[k*V:(k+1)*V, l*V:(l+1)*V] = np.array(coo_matrix( (map( lambda i : c[i], idx) , 
#                                                                       zip(*idx)), shape = (V, V)).todense())
                Ckl = coo_matrix( (list( map( lambda i : c[i], idx) ), list(zip(*idx)) ), shape = (V, V)).tolil()
                #C[int(k*V):int((k+1)*V),int( l*V ):int((l+1)*V) ] = Ckl.tocsr() 
                for i in range(V):
                    for j in range(len(Ckl.rows[i])):
                        C[int(k*V)+i, int(l*V) + Ckl.rows[i][j]] = Ckl.data[i][j]
                        C[int(l*V) + Ckl.rows[i][j], int(k*V)+i] = Ckl.data[i][j]


                pbar.update(k*K + l+1)
 
#        print("Checking symmetricity as a sanity check...") #can be memory hungry
#        isSymmetric = all((C == C.T).flatten()) 
#        if not isSymmetric:
#            print("Warning...matrix is not symmetric")
#        else:
#            print("symmetricity check passed")

        #np.savez("C", C)
        save_csr(csr_matrix(C), "C_{}_{}".format(IDX, CHU))
        save_csr(csr_matrix(A), "A")
        for i in range(len(MAXDs)):
            save_csr(csr_matrix(Ds[i]), "D{}_{}_{}".format(MAXDs[i],IDX, CHU))
            os.system("mv D{}_* {}/{}/".format(MAXDs[i], root_mem_path, memory_path))

        os.system("mv C_* {}/{}/".format(root_mem_path, memory_path))

        f = open("A_log.txt", "w")
        f.write("{} {}".format(A.shape[0], A.shape[1]))
        f.close()
        os.system("mv A* {}/{}/".format(root_mem_path, memory_path))

































































































































































































































































































































































































































































