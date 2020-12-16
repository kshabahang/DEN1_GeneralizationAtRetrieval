from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.stats import ttest_rel


conds = ['intact','lesioned']

#for cond in conds:
if True:
    cond = 'Goodman'
    fname = "output_goodman_etal_intact.pkl"#"PPR_VBP_2_PPRS_VBP_output_munt_and_hans_{}.pkl".format(cond)
#    nouns = "PPRS_NN_2_PPR_NN_output_munt_and_hans_{}.pkl".format(cond)
    
    f = open("../rsc/"+fname, "rb")
    goodman = pickle.load(f)
    f.close()
    
    #f = open("../rsc/"+nouns, "rb")
    #nouns = pickle.load(f)
    #f.close()
    
#    nouns_corr = np.array([nouns['correct']['vlens'][i][-1] for i in range(len(nouns['correct']['vlens']))])
#    nouns_incr = np.array([nouns['incorrect']['vlens'][i][-1] for i in range(len(nouns['incorrect']['vlens']))] )
    
    goodman_corr = np.array([goodman['correct']['vlens'][i][-1] for i in range(len(goodman['correct']['vlens']))])
    goodman_incr = np.array([goodman['incorrect']['vlens'][i][-1] for i in range(len(goodman['incorrect']['vlens']))])
    
    
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
    d = round(np.mean(goodman_corr - goodman_incr)/np.std(goodman_corr - goodman_incr), 2)
#    d_nouns = round(np.mean(nouns_corr - nouns_incr)/np.std(nouns_corr - nouns_incr), 2)

    acc =100*round( len(np.where(goodman_corr - goodman_incr > 0)[0]) / len(goodman_corr) , 2)
 #   acc_nouns =100*round( len(np.where(nouns_corr - nouns_incr > 0)[0]) / len(nouns_corr) , 2)

    
    fig = plt.figure()
    plt.title(cond)
    plt.hist(goodman_corr - goodman_incr, alpha = 0.5, label = "{} --- {}%".format(d, acc))
 #   plt.hist(nouns_corr - nouns_incr, alpha = 0.5, label = "Nouns: {} --- {}%".format(d_nouns, acc_nouns))
    plt.axvline(0)
    plt.legend()
    fig.show()

    N = 32
    print(N, ttest_rel(goodman_corr[:N], goodman_incr[:N]))
    N = 35
    print(N, ttest_rel(goodman_corr[:N], goodman_incr[:N]))
