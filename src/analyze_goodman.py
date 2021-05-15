from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import spacy

nlp =  spacy.load("en_core_web_lg")


conds = ['intact','lesioned']

#for cond in conds:
if True:
    cond = 'Goodman'
    fname = "output_goodman_etal_intact_0p55nege1.pkl"
    
    f = open(fname, "rb")
    goodman = pickle.load(f)
    f.close()
    

    
    goodman_corr = np.array([goodman['correct']['vlens'][i][-1][0] for i in range(len(goodman['correct']['vlens']))])
    goodman_incr = np.array([goodman['incorrect']['vlens'][i][-1][0] for i in range(len(goodman['incorrect']['vlens']))])



    vlens_verbs = []
    vlens_nouns = []
    for i in range(len(goodman['correct']['probe'])):
        bg = goodman['correct']['probe'][i]
        [w1,w2] = bg.split()
        pos1 = nlp(w1)[0].pos_
        pos2 = nlp(w2)[0].pos_
        if pos2 == "NOUN":
            vlens_nouns.append( goodman['correct']['vlens'][i][-1][0] )
        elif pos2 == "VERB":
            vlens_verbs.append( goodman['correct']['vlens'][i][-1][0] )

    ttest = ttest_ind(vlens_nouns, vlens_verbs)
    t = np.round(ttest[0],2)
    p = np.round(ttest[1],3)
    M1 = np.round(np.mean(vlens_nouns),5)
    M2 = np.round(np.mean(vlens_verbs),5)
    df= min([len(vlens_nouns)-1, len(vlens_verbs)-1])
    print("Mean(Noun): {} --- Mean(Verb): {} --- t({}) = {}, p = {}".format(M1, M2, df, t, p ))





    vlens_verbs = []
    vlens_nouns = []
    for i in range(len(goodman['incorrect']['probe'])):
        bg = goodman['incorrect']['probe'][i]
        [w1,w2] = bg.split()
        pos1 = nlp(w1)[0].pos_
        pos2 = nlp(w2)[0].pos_
        if pos2 == "NOUN":
            vlens_nouns.append( goodman['incorrect']['vlens'][i][-1][0] )
        elif pos2 == "VERB":
            vlens_verbs.append( goodman['incorrect']['vlens'][i][-1][0] )


    ttest = ttest_ind(vlens_nouns, vlens_verbs)
    t = np.round(ttest[0],2)
    p = np.round(ttest[1],3)
    M1 = np.round(np.mean(vlens_nouns),5)
    M2 = np.round(np.mean(vlens_verbs),5)
    df= min([len(vlens_nouns)-1, len(vlens_verbs)-1])
    print("Mean(Noun): {} --- Mean(Verb): {} --- t({}) = {}, p = {}".format(M1, M2, df, t, p ))



    diff = goodman_corr - goodman_incr

    df = pd.DataFrame(np.array([goodman_corr, goodman_incr, diff]).T, columns=["Congruent", "Incongruent", "Difference"])   



    d = np.mean(goodman_corr - goodman_incr)/np.std(goodman_corr - goodman_incr)


    sns.set_palette("Purples_r") 
    N = 32
    print(N, ttest_rel(goodman_corr[:N], goodman_incr[:N]))

    fig = plt.figure()
    #sns.distplot(df['Congruent'], kde=False, label="Congruent")
    #sns.distplot(df['Incongruent'], kde=False, label="Incongruent")
    plot = sns.distplot(df["Difference"], kde=False)
#    plt.legend()
    plt.axvline(0, color="red")
    plot.set_xlabel("Familiarity difference", fontsize=15)
    plot.set_ylabel("Frequency", fontsize=15)
    fig.show()
