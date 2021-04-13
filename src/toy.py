import numpy as np
from collections import Counteri
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def vcos(u,v):
    if u.dot(v) == 0:
        return 0
    else:
        return u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))



wvecs = {"the":[1,1,-1,-1], "cat":[-1,1,-1,1], "a":[1,1,1,1], "dog":[1,-1,-1,1]}
wvecs = {w:wvecs[w]/np.linalg.norm(wvecs[w]) for w in wvecs.keys()}

wvec_mat = np.vstack([wvecs['the'], wvecs['cat'], wvecs['a'], wvecs['dog']])
vocab = ['the','cat', 'a', 'dog']

p1 =  np.hstack([wvecs['the'], wvecs['cat']])
p2 =  np.hstack([wvecs['a'], wvecs['dog']])

W = 1.2*np.outer(p1, p1) + 1.1*np.outer(p2, p2)

probes = {"old_strong":p1,
            "old_weak":p2,
            "old_strong_part":np.hstack([wvecs['the'], np.zeros(4)]),
            "old_weak_part":np.hstack([wvecs['a'], np.zeros(4)]),
            "new1":np.hstack([wvecs['a'], wvecs['cat']]),
            "new2":np.hstack([wvecs['the'], wvecs['dog']]),
            "odd1":np.hstack([wvecs['cat'], wvecs['a']]),
            "odd2":np.hstack([wvecs['dog'], wvecs['the']])}


runDEN = True
runBSB = False

noise = 0.3

###Dynamic-Eigen-Net
if runDEN:
    print("Dynamic-Eigen-Net")
    for probe_type in probes.keys():
        resps = []
        for l in range(1000):
            probe = probes[probe_type] + np.random.normal(0, noise)
            probe /= np.linalg.norm(probe)
            
            x = 1*probe
            W_prime = W + np.outer(x, x)
            x_prime = probe.dot(W_prime)
            x_prime /= np.linalg.norm(x_prime)
            k = 1
            while(np.linalg.norm(x - x_prime) > 1e-7):
                x = 1*x_prime
                x_prime = x.dot(W_prime)
                x_prime /= np.linalg.norm(x_prime)
                k += 1
            
          
            s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
            s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
    
        
            w1 = vocab[np.argmax(s1)]
            w2 = vocab[np.argmax(s2)]
            
            resp = w1 + " " + w2
            resps.append(resp)
        
       
        counts =  Counter(resps)
        print(probe_type)
        for cue in counts.keys():
            print(cue, counts[cue]/1000)


###BSB
if runBSB:
    print('-'*32)
    print("Brain-State-in-a-Box")
    for probe_type in probes.keys():
        resps = []
        for l in range(1000):
            probe = probes[probe_type] + np.random.normal(0, noise)
            probe /= np.linalg.norm(probe)
    
            x1 = 1*probe
            x2 = x1.dot(W).clip(min=-1, max=1)
            k = 1
            while(np.linalg.norm(x1 - x2) > 1e-7):
                x1 = 1*x2
                x2 = x1.dot(W).clip(min=-1, max=1)
                k += 1
            
          
            s1 = np.array([np.abs(vcos(x2[:4], wvec_mat[j])) for j in range(len(vocab))])
            s2 = np.array([np.abs(vcos(x2[4:], wvec_mat[j])) for j in range(len(vocab))])
        
            w1 = vocab[np.argmax(s1)]
            w2 = vocab[np.argmax(s2)]
            
            resp = w1 + " " + w2
            resps.append(resp)
        

        counts =  Counter(resps)
        print(probe_type)
        for cue in counts.keys():
            print(cue, counts[cue]/1000)




###DEN plots
p_fine  = probes['new2']
p_odd   = probes['odd2']

frames = {"the dog":[], "dog the":[]}

probe = p_fine + np.random.normal(0, noise)
probe /= np.linalg.norm(probe)

frames["the dog"].append(probe)

x = 1*probe
W_prime = W + np.outer(x, x)
x_prime = probe.dot(W_prime)
x_prime /= np.linalg.norm(x_prime)
frames["the dog"].append(x_prime)
k = 1
while(np.linalg.norm(x - x_prime) > 1e-7):
    x = 1*x_prime
    x_prime = x.dot(W_prime)
    x_prime /= np.linalg.norm(x_prime)

    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])



    frames["the dog"].append(np.hstack([s1,s2]))
    k += 1

                                                                                   
s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
                                                                                   
                                                                                   
w1 = vocab[np.argmax(s1)]
w2 = vocab[np.argmax(s2)]

resp = w1 + " " + w2



frames_new = np.array(frames['the dog'])

probe = p_odd + np.random.normal(0, noise)
probe /= np.linalg.norm(probe)

frames["dog the"].append(probe)

x = 1*probe
W_prime = W + np.outer(x, x)
x_prime = probe.dot(W_prime)
x_prime /= np.linalg.norm(x_prime)
frames["dog the"].append(x_prime)
k = 1
while(np.linalg.norm(x - x_prime) > 1e-7):
    x = 1*x_prime
    x_prime = x.dot(W_prime)
    x_prime /= np.linalg.norm(x_prime)


    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])



    frames["dog the"].append(np.hstack([s1,s2]))
    k += 1

                                                                                   
s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
                                                                                   
                                                                                   
w1 = vocab[np.argmax(s1)]
w2 = vocab[np.argmax(s2)]

resp = w1 + " " + w2



frames_new = np.array(frames['the dog'])


activation = []
slot = []
word = []
iteration=[]
(n, m) = frames_new.shape 
for i in range(m):
    activation +=  list( frames_new[:, i] )
    if i < 4:
        slot += [1]*n
    else:
        slot += [2]*n
    iteration += range(n)
    word += [vocab[i%4]]*n

df = pd.DataFrame({"Activation": activation, "Word":word, "Slot": slot, "Iteration": iteration})

sns.set_theme(style="ticks")
palette = sns.color_palette("Set2")
plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", palette="dark", linewidth=4)
plot.set(ylim=(0,1))



frames_odd = np.array(frames['dog the'])


activation = []
slot = []
word = []
iteration=[]
(n, m) = frames_odd.shape 
for i in range(m):
    activation +=  list( frames_odd[:, i] )
    if i < 4:
        slot += [1]*n
    else:
        slot += [2]*n
    iteration += range(n)
    word += [vocab[i%4]]*n

df = pd.DataFrame({"Activation": activation, "Word":word, "Slot": slot, "Iteration": iteration})

sns.set_theme(style="ticks")
palette = sns.color_palette("Set2")
plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", palette="dark", linewidth=4)
plot.set(ylim=(0,1))
