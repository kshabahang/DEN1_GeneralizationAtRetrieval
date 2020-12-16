from matplotlib import pyplot as plt
import pickle
import numpy as np


conds = ['intact','lesioned']

for cond in conds:
    verbs = "PPR_VBP_2_PPRS_VBP_output_munt_and_hans_{}.pkl".format(cond)
    nouns = "PPRS_NN_2_PPR_NN_output_munt_and_hans_{}.pkl".format(cond)
    
    f = open("../rsc/"+verbs, "rb")
    verbs = pickle.load(f)
    f.close()
    
    f = open("../rsc/"+nouns, "rb")
    nouns = pickle.load(f)
    f.close()
    
    nouns_corr = np.array([nouns['correct']['vlens'][i][-1] for i in range(len(nouns['correct']['vlens']))])
    nouns_incr = np.array([nouns['incorrect']['vlens'][i][-1] for i in range(len(nouns['incorrect']['vlens']))] )
    
    verbs_corr = np.array([verbs['correct']['vlens'][i][-1] for i in range(len(verbs['correct']['vlens']))])
    verbs_incr = np.array([verbs['incorrect']['vlens'][i][-1] for i in range(len(verbs['incorrect']['vlens']))])
    
    
    #fig, [f1,f2] = plt.subplots(nrows=2, ncols=1)
    
    #f1.hist(verbs_corr, alpha=0.5, label = "Correct")
    #f1.hist(verbs_incr, alpha=0.5, label = "Incorrect")
    #f1.legend()
    #f1.set_title("Verbs")
    #
    #
    #
    #f2.hist(nouns_corr, alpha=0.5, label = "Correct")
    #f2.hist(nouns_incr, alpha=0.5, label = "Incorrect")
    #f2.legend()
    #f2.set_title("Nouns")
    
    
    
    #fig.show()
    d_verbs = round(np.mean(verbs_corr - verbs_incr)/np.std(verbs_corr - verbs_incr), 2)
    d_nouns = round(np.mean(nouns_corr - nouns_incr)/np.std(nouns_corr - nouns_incr), 2)

    acc_verbs =100*round( len(np.where(verbs_corr - verbs_incr > 0)[0]) / len(verbs_corr) , 2)
    acc_nouns =100*round( len(np.where(nouns_corr - nouns_incr > 0)[0]) / len(nouns_corr) , 2)

    
    fig = plt.figure()
    plt.title(cond)
    plt.hist(verbs_corr - verbs_incr, alpha = 0.5, label = "Verbs: {} --- {}%".format(d_verbs, acc_verbs))
    plt.hist(nouns_corr - nouns_incr, alpha = 0.5, label = "Nouns: {} --- {}%".format(d_nouns, acc_nouns))
    plt.axvline(0)
    plt.legend()
    fig.show()
