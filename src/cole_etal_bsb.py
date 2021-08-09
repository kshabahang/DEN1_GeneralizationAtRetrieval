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





def proj(a, v):
    return (a.dot(v)/v.dot(v))*v


if __name__ == "__main__":

    N =2**17#16360#2**16

    K = 2#2

    comp_idx = int(sys.argv[1])

    
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
               "feedback":"saturate",
               "init_weights":False,
               "gpu":False,
               "localist":False,
               "distributed":True,
               "maxiter":1000,
               "explicit":True,
               "sparse":False,
               "row_center":False,
               "col_center":False,
               "norm":"pmi"}
    ANet = AssociativeNet(hparams)

    memory_path ="FRENCH" #sys.argv[1]

    NSamples = 150

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


    
    #A = load_csr(root_mem_path + "/{}/A".format(memory_path),  A_shape, dtype=np.int32) #pre-computed



    ANet.A = None# A



    ANet.COUNTS = csr_matrix(C)



    del C
    #sys.exit()
    ##drop low freq term
    ANet.prune(min_wf = 70) #10



    if hparams["distributed"]:
        print("Constructing e-vecs")
        obv = OBVGen(14)
        #ANet.E = list(np.load(root_mem_path + "/{}/E{}.npy".format(memory_path, N)))
    
        ANet.E = obv.E[:ANet.V]/np.linalg.norm(obv.E[0])
    
    ANet.nullvec = np.zeros(ANet.N*ANet.K)



    #toLoad = True
    #if toLoad:
    #    print("Loading weight matrix")
    #    ANet.W = np.load(root_mem_path + "/{}/pmi.npy".format(memory_path))
    #    print("Loading eigenspectrum")
    #    ANet.ei = np.load(root_mem_path + "/{}/ei_pmi.npy".format(memory_path))
    #    ANet.ev = np.load(root_mem_path + "/{}/ev_pmi.npy".format(memory_path))
    #    ANet.W /= ANet.ei[0]
    #else:
    #    print("Crunching out the weights...")
    #    ANet.compute_weights(binaryMat=False)
    #    print("Saving weight matrix")
    #    np.save(root_mem_path + "/{}/pmi".format(memory_path), ANet.W )
    #    ANet.update_eig()
    #    print("Saving eigenspectrum")
    #    np.save(root_mem_path + "/{}/ei_pmi".format(memory_path), ANet.ei )
    #    np.save(root_mem_path + "/{}/ev_pmi".format(memory_path), ANet.ev )


    ANet.compute_weights(binaryMat=False)


    ANet.nullvec = np.zeros((K*ANet.V))
    ANet.N = ANet.V

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()
    

    ANet.norm_eig(verbos=True, eps=1e-8)
    ev = ANet.ev[:, 0]
    eta = 0.55
    for i in range(ANet.V*ANet.K):
        ANet.W[i, :] -= eta*ev[i]*ev*ANet.ei
    
    ANet.alpha = ANet.ei + 0.001*ANet.ei

    ANet.N = 1*N
    E = np.array(ANet.E).T
    
    W_old = 1*ANet.W
    W_new = np.zeros((ANet.N*ANet.K, ANet.N*ANet.K))
    for p in range(ANet.K):
        for q in range(ANet.K):
            W_new[p*ANet.N:(p+1)*ANet.N, q*ANet.N:(q+1)*ANet.N] = E.dot(ANet.W[p*ANet.V:(p+1)*ANet.V, q*ANet.V:(q+1)*ANet.V]).dot(E.T)
    
    ANet.W = W_new



    ANet.theta = 1

    toLesion = False


    f = open("cole_stims.pkl", "rb")
    stims = pickle.load(f)
    f.close()

    nouns = list(stims.keys())




    runSet = sys.argv[2] == "bgs"

    if runSet: 

        scores = {"correct_open":{}, "incorrect_number_open":{}, "incorrect_gender_open":{}, "correct_close":{}, "incorrect_number_close":{}, "incorrect_gender_close":{}}
        corr_lens = []
        incorr_lens=[]



        for k in range(len(nouns)):

            close_C = stims[nouns[k]]["close"]
            open_C = stims[nouns[k]]["open"]

            ws = list(close_C.values()) + list(open_C.values()) + [nouns[k]]

            if sum([ws[l] in ANet.vocab for l in range(len(ws))]) == len(ws): # we have all words

                #(frq, [correct, incorrect]) = pair_items[k]
                frq = 0
                correct_open = open_C['control'] + " " + nouns[k]
                incorrect_number_open = open_C['number'] + " " + nouns[k]
                incorrect_gender_open =  open_C['gender'] + " " + nouns[k]

                correct_close = close_C['control'] + " " + nouns[k] 
                incorrect_number_close = close_C['number'] + " " + nouns[k]
                incorrect_gender_close = close_C['gender'] + " " + nouns[k]
        
                bank_lbls = ['t-0', 't-1']
          
        
                labels_open = ["correct_open", "incorrect_number_open", "incorrect_gender_open"]
                labels_close =[ "correct_close", "incorrect_number_close", "incorrect_gender_close"]
                probes_open = [correct_open,incorrect_number_open,incorrect_gender_open]
                probes_close =[correct_close, incorrect_number_close, incorrect_gender_close]

                ###open first
                [w1_c, w2_c] = probes_open[0].split()

                if toLesion:
                    ANet - (w1_c, w2_c)
        
                for i in range(len(probes_open)):
                    probe = probes_open[i]
        
                    ANet.probe(probe)
        
                    terminal = ' '.join([ANet.banks[bank_lbls[j]][0][0] for j in range(len(bank_lbls))])
                    #change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
                    cycles = ANet.count
        
                    if "vlens" not in scores[labels_open[i]]:
                        scores[labels_open[i]]["vlens"] = [ANet.vlens]
                        #scores[labels[i]]["ncycles"] = [len(ANet.frames)]
                        scores[labels_open[i]]["probe"] = [probe]
                        scores[labels_open[i]]["freq"] = [frq]
                    else:
                        scores[labels_open[i]]["vlens"].append(ANet.vlens)
                        #scores[labels[i]]["ncycles"].append(len(ANet.frames))
                        scores[labels_open[i]]["probe"].append(probe)                
                        scores[labels_open[i]]["freq"].append(frq)
 
                if toLesion:
                    ~ ANet #reset

                ###close second


                [w1_c, w2_c] = probes_close[0].split()

                if toLesion:
                    ANet - (w1_c, w2_c)
        
                for i in range(len(probes_close)):
                    probe = probes_close[i]
        
                    ANet.probe(probe)
        
                    terminal = ' '.join([ANet.banks[bank_lbls[j]][0][0] for j in range(len(bank_lbls))])
                    #change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
                    cycles = ANet.count
        
                    if "vlens" not in scores[labels_close[i]]:
                        scores[labels_close[i]]["vlens"] = [ANet.vlens]
                        #scores[labels[i]]["ncycles"] = [len(ANet.frames)]
                        scores[labels_close[i]]["probe"] = [probe]
                        scores[labels_close[i]]["freq"] = [frq]
                    else:
                        scores[labels_close[i]]["vlens"].append(ANet.vlens)
                        #scores[labels[i]]["ncycles"].append(len(ANet.frames))
                        scores[labels_close[i]]["probe"].append(probe)                
                        scores[labels_close[i]]["freq"].append(frq)
 
                if toLesion:
                    ~ ANet #reset



        if toLesion:
            print( "Lesioned")
        else:
            print("Not lesioned")


        f = open(root_mem_path + "/{}/cole_intact.pkl".format(memory_path), "wb")
        pickle.dump(scores, f)
        f.close()




















































































































































































































































































































































































