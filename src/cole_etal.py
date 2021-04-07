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

    comp_idx = int(sys.argv[1])

    
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


    
    A = load_csr(root_mem_path + "/{}/A".format(memory_path),  A_shape, dtype=np.int32) #pre-computed


    #maxds =[5,10,15,20,50,100,200,300,500,1000]
    #maxds = [50]#[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    #Ds = []
    #for i in range(len(maxds)):
    #    MAXD = maxds[i]
    #    D = load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path, MAXD,0, totalChunks),  (V*K, V*K), dtype=np.int64) #pre-computed
    #    for IDX in range(1, nchunk):
    #        D[:, :] += load_csr(root_mem_path + "/{}/D{}_{}_{}".format(memory_path,MAXD, IDX, totalChunks), (V*K, V*K), dtype=np.int64)
    #    
    #    Ds.append( csr_matrix(D) )


    #ANet.Ds = Ds
    #ANet.D = ANet.Ds[0]
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
    ANet.prune(min_wf = 50) #10

    f = open("cole1.txt", "r")
    bg_pool = f.readlines()
    f.close()
    close_class = []
    open_class = []

    for i in range(len(bg_pool)):
        [dts, adjs, n] = bg_pool[i].split()
        [adj_sg, adj_pl] = adjs.split('/')
        [dt_sg, dt_pl] = dts.split('/')
        closed_control = dt_sg + " " + n 
        closed_number = dt_pl + " " + n
        open_control = adj_sg + " " + n
        open_number  = adj_pl + " " + n
        check = [adj_sg, adj_pl, dt_sg, dt_pl, n]
        isSafe = True
        for j in range(len(check)):
            if check[j] not in ANet.vocab:
                isSafe = False
        if isSafe:
            #print(closed_control, closed_number)
            #print(open_control, open_number)
            close_class.append([closed_control, closed_number])
            open_class.append([open_control, open_number])





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

    runSet = sys.argv[2] == "bgs"

    if runSet: 

        if comp_idx == 0:
            bgs = open_class
            pair_set = "ADJNOM_2_ADJSNOM"
        if comp_idx == 1:
            pair_set = "DETNOM_2_DETSNOM"
            bgs = close_class
        
        probes_g  = [] 
        probes_ug = []
        for l in range(len(bgs)):
            [corr, incorr] = bgs[l]
            [corrA, corrB] = corr.split()
            [incorrA, incorrB] = incorr.split()

            probes_g.append(corrA + " " + corrB)
            probes_ug.append(incorrA + " " + incorrB)


        scores = {"correct":{}, "incorrect":{}}
        corr_lens = []
        incorr_lens=[]



        for k in range(NSamples):
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
                        corr_lens.append(ANet.vlens[-1][0])
                    else:
                        incorr_lens.append(ANet.vlens[-1][0])
        
                    if "vlens" not in scores[labels[i]]:
                        scores[labels[i]]["vlens"] = [ANet.vlens[-1]]
                        #scores[labels[i]]["ncycles"] = [len(ANet.frames)]
                        scores[labels[i]]["probe"] = [probe]
                        scores[labels[i]]["freq"] = [frq]
                    else:
                        scores[labels[i]]["vlens"].append(ANet.vlens[-1])
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

        f = open(root_mem_path + "/"+pair_set + "_{}.pkl".format(memory_path), "wb")
        pickle.dump(scores, f)
        f.close()




















































































































































































































































































































































































