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
    fname = "output_goodman_etal_intact_0p55nege1.pkl"
    
    f = open(fname, "rb")
    goodman = pickle.load(f)
    f.close()
    

    
    goodman_corr = np.array([goodman['correct']['vlens'][i][-1][0] for i in range(len(goodman['correct']['vlens']))])
    goodman_incr = np.array([goodman['incorrect']['vlens'][i][-1][0] for i in range(len(goodman['incorrect']['vlens']))])

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
