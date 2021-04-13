import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

f = open("../rsc/cole_intact.pkl", "rb")
cole = pickle.load(f)
f.close()

iv_vocab = ["open", "close"]
iv_type   = ["correct", "incorrect_number", "incorrect_gender"]

DV1= []
DV2= []
IV1= []
IV2= []
for i in range(len(iv_vocab)):
    for j in range(len(iv_type)):
        key = iv_type[j]+"_"+iv_vocab[i] 
        v_1s = [cole[key]['vlens'][k][1][0] for k in range(len(cole[key]['vlens']))]
        v_ns = [cole[key]['vlens'][k][-1][0] for k in range(len(cole[key]['vlens']))]
        IV1 += [iv_vocab[i]]*len(v_ns)
        IV2 += [iv_type[j]]*len(v_ns) 
        DV1 += v_1s
        DV2 += v_ns

df_1 = pd.DataFrame({"Vocabulary":IV1, "Condition":IV2, "Familiarity":DV1})
df_n = pd.DataFrame({"Vocabulary":IV1, "Condition":IV2, "Familiarity":DV2})
subtractGrandMean = False
if subtractGrandMean:
    df_1["Familiarity"] -= df_1["Familiarity"].mean()
    df_n["Familiarity"] -= df_n["Familiarity"].mean()

scaleUp = False
if scaleUp:
    df_1["Familiarity"] *= 1000 
    df_n["Familiarity"] *= 1000

#sns.displot(df, x="Familiarity", hue="Vocabulary", col="Condition", row="Time")
#plt.show()
sns.barplot(x="Condition", y="Familiarity", hue="Vocabulary", data = df_n)
plt.show()


###collapse over gender and number
f_correct_close = df_n[(df_n['Condition'] == 'correct') * (df_n['Vocabulary'] == 'close')].Familiarity
f_incorrectGen_close= df_n[(df_n['Condition'] == 'incorrect_gender') * (df_n['Vocabulary'] == 'close')].Familiarity
f_incorrectNum_close=df_n[(df_n['Condition'] == 'incorrect_number') * (df_n['Vocabulary'] == 'close')].Familiarity

d_num_c = np.array(f_correct_close) - np.array(f_incorrectNum_close)
disc_num_c = np.mean(d_num_c)/np.std(d_num_c)

d_gen_c = np.array(f_correct_close) - np.array(f_incorrectGen_close)
disc_gen_c = np.mean(d_gen_c)/np.std(d_gen_c)

f_correct_open = df_n[(df_n['Condition'] == 'correct') * (df_n['Vocabulary'] == 'open')].Familiarity
f_incorrectGen_open= df_n[(df_n['Condition'] == 'incorrect_gender') * (df_n['Vocabulary'] == 'open')].Familiarity
f_incorrectNum_open=df_n[(df_n['Condition'] == 'incorrect_number') * (df_n['Vocabulary'] == 'open')].Familiarity

d_num_o = np.array(f_correct_open) - np.array(f_incorrectNum_open)
disc_num_o = np.mean(d_num_o)/np.std(d_num_o)

d_gen_o = np.array(f_correct_open) - np.array(f_incorrectGen_open)
disc_gen_o = np.mean(d_gen_o)/np.std(d_gen_o)

N = len(d_num_o)

