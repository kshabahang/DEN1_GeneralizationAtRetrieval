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

    N =16360#2**16

    K = 2

    

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

    memory_path = "TASA"#sys.argv[1]

#    sweep_idx_i = int(sys.argv[2])
#    sweep_idx_j = int(sys.argv[3])



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

    ###take out any counts less than a criterion
    #min_cc = 2
    #for i in range(len(C.data)):
    #    if C.data[i] < min_cc:
    #        C.data[i] = 0
    #C.eliminate_zeros()

    ANet.A= None



    ANet.COUNTS = csr_matrix(C)
    del C
    print("Crunching out the weights...")
    #ANet.compute_weights()

    ANet.prune(min_wf = 50)
    ANet.W = np.load(root_mem_path + "/{}/pmi.npy".format(memory_path))
    ANet.ei = np.load(root_mem_path + "/{}/ei_pmi.npy".format(memory_path))
    ANet.ev = np.load(root_mem_path + "/{}/ev_pmi.npy".format(memory_path))
    ANet.W /= max(ANet.ei)

    ANet.nullvec = np.zeros((K*ANet.V))
    ANet.N = ANet.V

    if ANet.hparams["gpu"]:
        ANet.nullvec = ANet.nullvec.cuda()


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


            f = open(root_mem_path + "/"+"output_goodman_etal_intact_0p55nege1.pkl", "wb")
            pickle.dump(scores, f)
            f.close()



















































































































































































































































































































































































