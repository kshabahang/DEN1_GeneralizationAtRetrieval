from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.stats import ttest_rel
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


conds = ['intact','lesioned']

#for cond in conds:
if True:
    cond = 'Goodman'
    fname = "output_goodman_etal_intact.pkl"
    
    f = open(fname, "rb")
    goodman = pickle.load(f)
    f.close()
    

    
    goodman_corr = np.array([goodman['correct']['vlens'][i][-1][0] for i in range(len(goodman['correct']['vlens']))])
    goodman_incr = np.array([goodman['incorrect']['vlens'][i][-1][0] for i in range(len(goodman['incorrect']['vlens']))])

    diff = goodman_corr - goodman_incr

    df = pd.DataFrame(np.array([goodman_corr, goodman_incr, diff]).T, columns=["Congruent", "Incongruent", "Difference"])   



    d = np.mean(goodman_corr - goodman_incr)/np.std(goodman_corr - goodman_incr)


    sns.set_palette("mako") 
    N = 32
    print(N, ttest_rel(goodman_corr[:N], goodman_incr[:N]))

    fig = plt.figure()
    #sns.distplot(df['Congruent'], kde=False, label="Congruent")
    #sns.distplot(df['Incongruent'], kde=False, label="Incongruent")
    sns.distplot(df["Difference"], kde=False)
#    plt.legend()
    plt.axvline(0)
    plt.xlabel("Familiarity(Congruent) - Familiarity(Incongruent)")
    plt.ylabel("Frequency")
    fig.show()
