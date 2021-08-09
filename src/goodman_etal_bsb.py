import sys, os
#from AssociativeNet import *
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

def sample_stims(lists):
    #pick 20 syntactical pairs from one of the lists
    list_idx = np.random.randint(2)
    idxs_leftA = [i for i in range(10)]
    idxs_leftB = [i for i in range(10)]
    np.random.shuffle(idxs_leftA)
    np.random.shuffle(idxs_leftB)

    idxs_rightA = [i for i in range(10)]
    idxs_rightB = [i for i in range(10)]
    np.random.shuffle(idxs_rightA)
    np.random.shuffle(idxs_rightB)

    incorrects = []
    corrects   = []

    for i in range(len(idxs_leftA)):
        if np.random.randint(2) == 1:
            idx_i = idxs_leftA[i]
            idx_j = idxs_rightA[i]
            w1 = lists[list_idx]['A_left'][idx_i]
            w2 = lists[list_idx]['A_right'][idx_j]
            correct = w1 + " " + w2
            if list_idx == 0:
                w3 = lists[1]['B_right'][idx_j]
            else:
                w3 = lists[0]['B_right'][idx_j]
            incorrect = w1 + " " + w3
        else:
            idx_i = idxs_leftA[i]
            idx_j = idxs_rightA[i]
            w1 = lists[list_idx]['B_left'][idx_i]
            w2 = lists[list_idx]['B_right'][idx_j]
            correct = w1 + " " + w2
            if list_idx == 0:
                w3 = lists[1]['A_right'][idx_j]
            else:
                w3 = lists[0]['A_right'][idx_j]
            incorrect = w1 + " " + w3

#        print(correct , incorrect)
        corrects.append(correct)
        incorrects.append(incorrect)

    return corrects, incorrects


def proj(a, v):
    return (a.dot(v)/v.dot(v))*v


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
               "feedback":"saturate",
               "init_weights":False,
               "gpu":False,
               "localist":False,
               "distributed":True,
               "maxiter":100,
               "explicit":True,
               "sparse":False,
               "row_center":False,
               "col_center":False,
               "norm":"pmi"}
    ANet = AssociativeNet(hparams)
    
    memory_path = "TASA"#sys.argv[1]
    
 
    
    
    NSamples = 20
    
    root_mem_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc"
    
    f = open(root_mem_path + "/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}
    ANet.V = len(ANet.vocab)
    
    #comp_idx = int(sys.argv[1])
    
    
    
    
    
    
    
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
    
        ANet.E = obv.E[:ANet.V]/np.linalg.norm(obv.E[0])
    
    ANet.nullvec = np.zeros(ANet.N*ANet.K)
    
    
    
    
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
        #ANet.update_eig()
        #print("Saving eigenspectrum")
        #np.save(root_mem_path + "/{}/ei_pmi".format(memory_path), ANet.ei )
        #np.save(root_mem_path + "/{}/ev_pmi".format(memory_path), ANet.ev )
    
    
    
    #pbar = ProgressBar(maxval=10).start()
    #W_new = np.zeros((2*ANet.E[0].shape[0],2*ANet.E[0].shape[0]))
    #for i in range(10):        
    #    for j in range(10):
    #        x_in = np.hstack([ANet.E[i], ANet.E[j]])
    #        W_new += ANet.W[i, j]*np.outer(x_in, x_in)
    
    #pbar.update(i+1)
    
    N = 1*ANet.N
    ANet.N = ANet.V
    ANet.norm_eig(verbos=True, eps=1e-5)
    ANet.W /= ANet.ei
    ev = ANet.ev[:, 0]
    eta = 0.55
    for i in range(ANet.N*ANet.K):
        ANet.W[i, :] -= eta*ev[i]*ev
    
    ANet.W *= ANet.ei
    ANet.alpha = ANet.ei + ANet.ei*0.001
    
    ANet.N = 1*N
    E = np.array(ANet.E).T
    
    W_old = 1*ANet.W
    W_new = np.zeros((ANet.N*ANet.K, ANet.N*ANet.K))
    for p in range(ANet.K):
        for q in range(ANet.K):
            W_new[p*ANet.N:(p+1)*ANet.N, q*ANet.N:(q+1)*ANet.N] = E.dot(ANet.W[p*ANet.V:(p+1)*ANet.V, q*ANet.V:(q+1)*ANet.V]).dot(E.T)
    
    ANet.W = W_new
    #ANet.update_eig()
    ANet.theta = 1


    grammatical = "VB_RBR PPRS_NN IN_VBG NNS_VBP NN_VBZ DT_NN JJ_NN NN_IN PPR_VBP".split()
    ungrammatical="RBR_VB PPR_NN IN_VBP NN_VBP NN_VBP NN_DT NN_JJ IN_NN PPRS_VBP".split()


    list1Aleft = "whose their our no the her my your any a".split()
    list1Aright= "planet corner night flower enemy fund turkey oven kid thing".split()
    list1Bleft = "you they people men it we she he i kids".split()
    list1Bright= "slid broke waved swear tied paid kissed sent froze play".split()

    L1 = {"A_left":list1Aleft , "A_right": list1Aright, "B_left":list1Bleft ,"B_right":list1Bright}

    list2Aleft = "your no a their our the any whose my her".split()
    list2Aright= "power bread rifle son edge tree road mud wife women".split()
    list2Bleft = "men they he it we you kids i people she".split()
    list2Bright= "led sprang exists drove agreed slept rated woke lit smiled".split()
    L2 = {"A_left":list2Aleft , "A_right": list2Aright, "B_left":list2Bleft ,"B_right":list2Bright}

    lists = [L1, L2]

    N = 35

    grammatical = []
    ungrammatical = []
    for n in range(N):
        corrects, incorrects = sample_stims(lists)
        grammatical += corrects
        ungrammatical += incorrects


    toLesion = False#True

    if True:
        scores = {"correct":{}, "incorrect":{}}

        for k in range(len(grammatical)):
            correct = grammatical[k]

            incorrect = ungrammatical[k]

            print (correct, incorrect)
        
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
                #change = round(np.linalg.norm(ANet.frames[0] - ANet.frames[-1]), 3)
                #cycles = len(ANet.frames)
        
                if "vlens" not in scores[labels[i]]:
                    scores[labels[i]]["vlens"] = [ANet.vlens]
                    scores[labels[i]]["ncycles"] = [ANet.count]
                  #  scores[labels[i]]["change"] = [change]
                  #  scores[labels[i]]["terminal"] = [terminal]
                    scores[labels[i]]["probe"] = [probe]
                    #scores[labels[i]]["freq"] = [frq]
                  #  scores[labels[i]]["frames"] = [ANet.frames]
                else:
                    scores[labels[i]]["vlens"].append(ANet.vlens)
                    scores[labels[i]]["ncycles"].append(ANet.count)
                   # scores[labels[i]]["change"].append(change)
                   # scores[labels[i]]["terminal"].append(terminal)
                    scores[labels[i]]["probe"].append(probe)                
                    #scores[labels[i]]["freq"].append(frq)
                  #  scores[labels[i]]["frames"].append(ANet.frames)
        
        
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

            f = open(root_mem_path + "/"+"output_goodman_etal_lesioned.pkl", "wb")
            pickle.dump(scores, f)
            f.close()


        else:
            print("Not lesioned")


            f = open(root_mem_path + "/"+"output_goodman_etal_intact_bsb.pkl", "wb")
            pickle.dump(scores, f)
            f.close()



















































































































































































































































































































































































