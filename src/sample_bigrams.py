import numpy as np
import pickle
import inflect
import os
from nltk.stem import WordNetLemmatizer

from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

#from corenlp import StanfordNLP 

def save_csr(matrix, filename):
    matrix.data.tofile(filename + "_"+ str(type(matrix.data[0])).split(".")[-1].strip("\'>") + ".data")
    matrix.indptr.tofile(filename + "_"+ str(type(matrix.indptr[0])).split(".")[-1].strip("\'>") + ".indptr")
    matrix.indices.tofile(filename + "_" +str(type(matrix.indices[0])).split(".")[-1].strip("\'>") + ".indices")

def load_csr(fpath, shape, dtype=np.float64, itype=np.int32):
    data = np.fromfile(fpath + "_{}.data".format(str(dtype).split(".")[-1].strip("\'>")), dtype)
    indptr = np.fromfile(fpath + "_{}.indptr".format(str(itype).split(".")[-1].strip("\'>")), itype)
    indices = np.fromfile(fpath + "_{}.indices".format(str(itype).split(".")[-1].strip("\'>")), itype)

    return csr_matrix((data, indices, indptr), shape = shape)

if __name__ == "__main__":



    
    root_path = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/"

    N =2**14

    K = 2

    memory_path = "TASA"
    sweep_idx = 0
    nchunk = 20 # number of chunks to combine into count matrix
    totalChunks = 20 #total number of chunks available








    f = open(root_path + "rsc/{}/vocab.txt".format(memory_path), "r")
    vocab = f.readlines()
    vocab = [vocab[i].strip() for i in range(len(vocab))]
    f.close()
    I = {vocab[i]:i for i in range(len(vocab))}

    f = open("log.out", "w")
    f.write("Let's go..." +"\n")
    f.close()


    V = len(vocab)

    

    toPOStag = False
    if toPOStag:
        f = open(root_path + "rsc/{}/ngram_freqs.pkl".format(memory_path), "rb")
        ngram_freqs= pickle.load(f)
        f.close()





        sNLP = StanfordNLP()
        bigram_by_POS = {}
    
        for bigram in ngram_freqs.keys():
            bg_freq = ngram_freqs[bigram]
            pos = sNLP.pos(bigram)
            pos1 = pos[0][1]
            pos2 = pos[1][1]
            key = pos1 + "_" +  pos2
            print (bigram, key)
            if key in bigram_by_POS:
                bigram_by_POS[key].append((bg_freq, bigram))
            else:
                bigram_by_POS[key] = [(bg_freq, bigram)]
    
        for pos_tag in bigram_by_POS.keys():
            bigram_by_POS[pos_tag] = sorted(bigram_by_POS[pos_tag])[::-1]
    
        f = open("bigram_by_POS.pkl", "wb")
        pickle.dump(bigram_by_POS, f)
        f.close()
    else:
        f = open(root_path + "rsc/{}/bigram_by_pos.pkl".format(memory_path), "rb")
        bigram_by_POS = pickle.load(f)
        f.close()

    getCounts = False
    if getCounts:
        tags = bigram_by_POS.keys()
        n_by_tag = []
        tag_and_rev= []
        for i in range(len(tags)):
            n_by_tag.append((len(bigram_by_POS[tags[i]]), tags[i]))
            [w1, w2]= tags[i].split()
            rev = w2 + " " + w1
            if rev in tags: 
                tag_and_rev.append([tags[i], len(bigram_by_POS[tags[i]]), len(bigram_by_POS[rev])])
    
        n_by_tag = sorted(n_by_tag)[::-1]

    sampleBGs = False ### DT NN - NN DT
    if sampleBGs:
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        bigram_types = ["DT NN"]
        bigrams = []
        pairs = []
        for bg_type in bigram_types:
            for bg in bigram_by_POS[bg_type]:
                [w1, w2] = bg[1].split()
                if w1 in vocab and w2 in vocab and 'xx' not in bg[1] and w1 != 'u' and w2 != 'u':
#                    freq_intact = C[I[w1], V + I[w2]]
#                    freq_revers = C[I[w2], V + I[w1]]
    #                bigrams.append([bg[1], freq_intact, freq_revers])
#                    if freq_revers == 0:
                    #print w1 + " " + w2, freq_intact, w2 + " " + w1, freq_revers
#                    bigrams.append((freq_intact, bg[1]))
                    pairs.append((bg[0], [w1 + " " + w2, w2 + " " + w1]))
#        bigrams = sorted(bigrams)[::-1]
#        f = open("bigrams_DT_NN.pkl","wb")
#        pickle.dump(bigrams[:50], f)
#        f.close()
        f = open("DT_NN_2_NN_DT.pkl", "wb") 
        pickle.dump(pairs, f)
        f.close()

    toPluralize = False ### NN VBZ - NN VBP
    if toPluralize:
        pairs = []
        p = inflect.engine()
        bgs = bigram_by_POS["NN VBZ"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1,w2] = bgs[i][1].split()
                w2_plr = p.plural(w2)
                if w1 in vocab and w2 in vocab and w2_plr in vocab:
                    #if C[I[w1], V+ I[w2_plr]] == 0:
#                    print w1 + " " + w2, "   ", w1 + " " + w2_plr
                     pairs.append((bgs[i][0], [w1 + " " + w2, w1 + " " + w2_plr ]))

        f = open("NN_VBZ_2_NN_VBP.pkl", "wb") 
        pickle.dump(pairs, f)
        f.close()


    toSingularize = False### NNS VBP - NN VBP
    if toSingularize:
        pairs = []
        lemmatizer = WordNetLemmatizer()
        bgs = bigram_by_POS["NNS VBP"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1,w2] = bgs[i][1].split()
                w1_sglr = lemmatizer.lemmatize(w1)
                if w1 in vocab and w2 in vocab and w1_sglr in vocab:
                    #if C[I[w1_sglr], V + I[w2]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w1_sglr + " " + w2]))

        f = open("NNS_VBP_2_NN_VBP.pkl", "wb") 
        pickle.dump(pairs, f)
        f.close()

    
    toUnING = False ### IN VBG - IN VBP
    if toUnING:
        pairs = []
        bgs = bigram_by_POS["IN VBG"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1, w2] = bgs[i][1].split()
                w2_noing = w2.replace("ing", "")
                if w1 in vocab and w2 in vocab and w2_noing in vocab:
#                    if C[I[w1], V + I[w2_noing]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w1 + " " + w2_noing]))

        f = open("IN_VBG_2_IN_VBP.pkl", "wb") 
        pickle.dump(pairs, f)
        f.close()



    toPossesive2Personal = False### PPR$ NN - PPR NN
    if toPossesive2Personal:
        pairs = []
        mapping = {"his":"he", "your":"you", "her":"she", "their":"them", "our":"us", "my":"me", "its":"it", "thy":"thee"}
        bgs = bigram_by_POS["PRP$ NN"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1,w2] = bgs[i][1].split()
                w1_ppr = mapping[w1] 
                if w1 in vocab and w2 in vocab:
                    #if C[I[w1_ppr], V + I[w2]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w1_ppr + " " + w2]))
        
        f = open("PPRS_NN_2_PPR_NN.pkl", "wb") 
        pickle.dump(pairs, f)
        f.close()

    toPersonal2Possessive = False
    if toPersonal2Possessive:
        pairs = []
        mapping = {"thou":"thy", "themselves":"their", "him":"his", "he":"his", "you":"your","she":"her", "them":"their","us":"our","me":"my", "it":"its", "thee":"thy", "we":"ours", "they":"their"}
        bgs = bigram_by_POS["PRP VBP"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1,w2] = bgs[i][1].split()
                if w1 in mapping:
                    w1_poss = mapping[w1]
                    if w1 in vocab and w2 in vocab:
                        #if C[I[w1_poss], V + I[w2]] == 0:
                         pairs.append((bgs[i][0], [w1 + " " + w2, w1_poss + " " + w2]))
        f = open("PPR_VBP_2_PPRS_VBP.pkl", "wb")
        pickle.dump(pairs, f)
        f.close()

    toFlipVB_RBR = False### VB RBR - RBR VB
    if toFlipVB_RBR:
        pairs = []
        bgs = bigram_by_POS["VB RBR"]
#        C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed
        for i in range(len(bgs)):
            if "xx" not in bgs[i][1]:
                [w1,w2] = bgs[i][1].split()
                if w1 in vocab and w2 in vocab:
                    #if C[I[w2], V + I[w1]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w2 + " " + w1]))

        f = open("VB_RBR_2_RBR_VB.pkl", "wb")
        pickle.dump(pairs, f)
        f.close()

    toFlipJJ_NN = False
    if toFlipJJ_NN:
         f = open("filter_bigrams.txt", "r")
         filter_out = f.readlines()
         f.close()
         filter_bgs = {}
         for i in range(len(filter_out)):
             [bg_type, odd] = filter_out[i].strip("\n").split(":")
             filter_bgs[bg_type] = map(lambda s : s.strip(), odd.split(","))

         pairs = []                                                                                      
         bgs = bigram_by_POS["JJ NN"]
#         C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed   
         for i in range(len(bgs)):
             if "xx" not in bgs[i][1] and bgs[i][1] not in filter_bgs["JJ_NN_2_NN_JJ"]:
                 [w1,w2] = bgs[i][1].split()
                 if w1 in vocab and w2 in vocab:
                     #if C[I[w2], V + I[w1]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w2 + " " + w1]))
 
         f = open("JJ_NN_2_NN_JJ.pkl", "wb")
         pickle.dump(pairs, f)
         f.close()

    toFlipNN_IN = False
    if toFlipNN_IN:
         pairs = []                                                                                      
         bgs = bigram_by_POS["NN IN"]
#         C = load_csr("../rsc/{}/C".format(memory_path, 0, totalChunks), (V*K, V*K)) #pre-computed   
         for i in range(len(bgs)):
             if "xx" not in bgs[i][1]:
                 [w1,w2] = bgs[i][1].split()
                 if w1 in vocab and w2 in vocab:
                     #if C[I[w2], V + I[w1]] == 0:
                     pairs.append((bgs[i][0], [w1 + " " + w2, w2 + " " + w1]))
 
         f = open("NN_IN_2_IN_NN.pkl", "wb")
         pickle.dump(pairs, f)
         f.close()


#    from matplotlib import pyplot as plt
#    plt.ion()
#    fig = plt.figure()
#    n = 0
#    tags = []
#    for i in range(len(n_by_tag)):
#        if n_by_tag[i][0] > 10000:
#            tag = n_by_tag[i][1]
#            lgfreqs = np.log(np.array(bigram_by_POS[tag])[:, 0].astype(int))
#            if lgfreqs[1000] >= 1.5:
#                plt.plot(lgfreqs, label = tag, alpha=0.5)
#                print tag
#                tags.append(tag)
#    fig.show()

    closedOpenTest = False
    if closedOpenTest:
        f = open("JJ_NN_2_NN_JJ.pkl", "rb")
        JJNN = pickle.load(f)
        f.close()
        f = open("DT_NN_2_NN_DT.pkl", "rb")
        DTNN = pickle.load(f)
        f.close()
        matched = []
        for i in range(len(DTNN)):
            [DT, NN1] = DTNN[i][1][0].split()
            fr1 = DTNN[i][0]
            for j in range(len(JJNN)):
                fr2 = JJNN[j][0]
                if fr1 == fr2:   
                    [JJ, NN2] = JJNN[j][1][0].split()
                    if NN1 == NN2:
                        print (DTNN[i], JJNN[j])
                        matched.append([DTNN[i], JJNN[j]])

    countBGs = False
    if countBGs:
        f = open("tasa_postagged.pkl", "rb")
        tasa = pickle.load(f)
        f.close()
        bgs     = {}
        bg_tags = {}
        for i in range(len(tasa)): #for each sentence
            if len(tasa[i]) == 2:
                sent = tasa[i][0]
                tags  = tasa[i][1]
                for j in range(len(sent)-1):
                    if len(sent[j:j+2]) == 2:
                        bg = ' '.join(sent[j:j+2])
                        if bg in bgs:
                            bgs[bg] += 1
                        else:
                            bgs[bg] = 1
                        bg_tag = ' '.join(tags[j:j+2])
                        if bg_tag in bg_tags:
                            if bg not in bg_tags[bg_tag]:
                                bg_tags[bg_tag].append(bg)
                        else:
                            bg_tags[bg_tag] = [bg]
        f = open("bigram_freqs.pkl", "wb")
        pickle.dump(bgs, f)
        f.close()
        
        for tag in bg_tags.keys():
            for i in range(len(bg_tags[tag])):
                bg_tags[tag][i] = (bgs[bg_tags[tag][i]], bg_tags[tag][i])

        for tag in bg_tags.keys():
            bg_tags[tag] = sorted(bg_tags[tag])[::-1]


        f = open("bigram_by_pos.pkl", "wb")
        pickle.dump(bg_tags, f)
        f.close()
    

    toCheck = False
    if toCheck:
        for fname in os.listdir("."):
            if "_2_" in fname and "pkl" in fname and "output" not in fname and "matched" not in fname:
                f = open(fname, "rb")
                bgs = pickle.load(f)
                f.close()
                f = open(fname.replace("pkl", "txt"), "w")
                for i in range(len(bgs)):
                    f.write(" : ".join(bgs[i][1]) + "\n")
                f.close()


