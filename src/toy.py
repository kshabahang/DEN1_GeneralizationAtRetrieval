import numpy as np
from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def vcos(u,v):
    if u.dot(v) == 0:
        return 0
    else:
        return u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))






vectors = [[1,1,-1,-1], [-1,1,-1,1], [1,1,1,1], [1,-1,-1,1]]


#wvecs = {"the":[1,1,-1,-1], "cat":[-1,1,-1,1], "a":[1,1,1,1], "dog":[1,-1,-1,1]}
wvecs = {"the":[1,1,1,1], "cat":[-1,1,-1,1], "a":[1,-1,-1,1], "dog":[1,1,-1,-1]}
vocab = ['the','cat', 'a', 'dog']

NIterProbe = 1000
NIterEncode=1


resps = {}


for m in range(NIterEncode):
    #word_assignment = list(range(len(vocab)))
    #np.random.shuffle(word_assignment)
    
    #wvecs = {vocab[word_assignment[i]]:vectors[i] for i in range(len(vocab))}
    #for j in range(len(vocab)):
    #    print(vocab[j], wvecs[vocab[j]])
    
    #wvecs = {w:wvecs[w]/np.linalg.norm(wvecs[w]) for w in wvecs.keys()}
    
    wvec_mat = np.vstack([wvecs['the'], wvecs['cat'], wvecs['a'], wvecs['dog']])
   

    
    p1 =  np.hstack([wvecs['the'], wvecs['cat']])
    p2 =  np.hstack([wvecs['a'], wvecs['dog']])
    p1 = p1/np.linalg.norm(p1)
    p2 = p2/np.linalg.norm(p2)
    
    W = 1.2*np.outer(p1, p1) + 1.1*np.outer(p2, p2)
    
    probes = {"old_strong":p1,
                "old_weak":p2,
                "old_strong_part":np.hstack([wvecs['the'], np.zeros(4)]),
                "old_weak_part":np.hstack([wvecs['a'], np.zeros(4)]),
                "new1":np.hstack([wvecs['the'], wvecs['dog']]),
                "new2":np.hstack([wvecs['a'], wvecs['cat']]),
                "odd1":np.hstack([wvecs['dog'], wvecs['the']]),
                "odd2":np.hstack([wvecs['cat'], wvecs['a']])}
    
    ei, ev = np.linalg.eig(W) 

    isort = np.argsort(ei)[::-1][:2]
    print(ei[isort])
    print(ev[:, isort])

    runDEN = True
    runBSB = True
    
    noise = 0.3
    
    ###Dynamic-Eigen-Net
    if runDEN:
        print("Dynamic-Eigen-Net")
        for probe_type in probes.keys():
            resps = []
            for l in range(NIterProbe):
                loadings = []
                probe = probes[probe_type] + np.random.normal(0, noise)
                probe /= np.linalg.norm(probe)
                
                x = 1*probe
                W_prime = W + np.outer(x, x)
                x_prime = probe.dot(W_prime)
                x_prime /= np.linalg.norm(x_prime)
                loadings.append(x_prime.dot(ev[isort].T))
                sts = []
                states = []
                k = 1
                while(np.linalg.norm(x - x_prime) > 1e-7):
                    x = 1*x_prime
                    x_prime = x.dot(W_prime)
                    x_prime /= np.linalg.norm(x_prime)
                    k += 1
                    loadings.append(x_prime.dot(ev[isort].T))        

                    s1 = np.array([vcos(x_prime[:4], wvec_mat[j]) for j in range(len(vocab))])
                    s2 = np.array([vcos(x_prime[4:], wvec_mat[j]) for j in range(len(vocab))])
                    sts.append([s1,s2])
                    states.append(x_prime)

                
                s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
                s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
        
                
                w1 = vocab[np.argmax(s1)]
                w2 = vocab[np.argmax(s2)]
                
                resp = w1 + " " + w2
                resps.append(resp)
                
                #if probe_type not in resps:
                #    resps[probe_type] = [resp]
                #else:
                #    resps[probe_type].append(resp)
            counts =  Counter(resps)
            print(probe_type)
            for cue in counts.keys():
                print(cue, counts[cue]/NIterProbe)
    
    
    ###BSB
    if runBSB:
        print('-'*32)
        print("Brain-State-in-a-Box")
        for probe_type in probes.keys():
            resps = []
            for l in range(NIterProbe):
                probe = probes[probe_type] + np.random.normal(0, noise)
                probe /= np.linalg.norm(probe)
        
                x1 = 1*probe
                x2 = (x1.dot(W) + probe)#.clip(min=-1, max=1)
                x2 /= np.linalg.norm(x2)
                k = 1
                while(np.linalg.norm(x1 - x2) > 1e-7):
                    x1 = 1*x2
                    x2 = (x1.dot(W) + probe)#.clip(min=-1, max=1)
                    x2 /= np.linalg.norm(x2)
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
                print(cue, counts[cue]/NIterProbe)


toPlot = True#False
if toPlot:
    ###DEN plots
    p_fine  = probes['new2']
    p_odd   = probes['odd2']
    fam_new = []
    frames = {"a cat":[], "cat a":[]}
    
    probe = p_fine + np.random.normal(0, noise)
    probe /= np.linalg.norm(probe)
    x = 1*probe
    W_prime = W + np.outer(x, x)
    x_prime = probe.dot(W_prime)
    fam_new.append(np.linalg.norm(x_prime))
    x_prime /= np.linalg.norm(x_prime)
    
    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
    
    frames["a cat"].append(np.hstack([s1,s2]))
    
    
    k = 1
    while(np.linalg.norm(x - x_prime) > 1e-7):
        x = 1*x_prime
        x_prime = x.dot(W_prime)
        fam_new.append(np.linalg.norm(x_prime))
        x_prime /= np.linalg.norm(x_prime)
    
        s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
        s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
    
    
    
        frames["a cat"].append(np.hstack([s1,s2]))
        k += 1
    
                                                                                       
    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
                                                                                       
                                                                                       
    w1 = vocab[np.argmax(s1)]
    w2 = vocab[np.argmax(s2)]
    
    resp = w1 + " " + w2
    
    
    
    frames_new = np.array(frames['a cat'])
    fam_odd = []
    probe = p_odd + np.random.normal(0, noise)
    probe /= np.linalg.norm(probe)
    x = 1*probe
    W_prime = W + np.outer(x, x)
    x_prime = probe.dot(W_prime)
    fam_odd.append(np.linalg.norm(x_prime))
    x_prime /= np.linalg.norm(x_prime)
    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
    
    frames["cat a"].append(np.hstack([s1,s2]))
    
    k = 1
    while(np.linalg.norm(x - x_prime) > 1e-7):
        x = 1*x_prime
        x_prime = x.dot(W_prime)
        fam_odd.append(np.linalg.norm(x_prime))
        x_prime /= np.linalg.norm(x_prime)
    
    
        s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
        s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
   
        #s1 = np.array([vcos(x_prime[:4], wvec_mat[j]) for j in range(len(vocab))])
        #s2 = np.array([vcos(x_prime[4:], wvec_mat[j]) for j in range(len(vocab))])
    
    
        frames["cat a"].append(np.hstack([s1,s2]))
        k += 1
    
                                                                                       
    s1 = np.array([np.abs(vcos(x_prime[:4], wvec_mat[j])) for j in range(len(vocab))])
    s2 = np.array([np.abs(vcos(x_prime[4:], wvec_mat[j])) for j in range(len(vocab))])
                                                                                       
                                                                                       
    w1 = vocab[np.argmax(s1)]
    w2 = vocab[np.argmax(s2)]
    
    resp = w1 + " " + w2
    
    frames_new = np.array(frames['a cat'])
    frames_odd = np.array(frames['cat a'])
    
    
    
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
   

    sns.set(font_scale=5)
    sns.set_theme(style="ticks")
    palette = sns.color_palette("Set2")
    plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", palette="dark", linewidth=8)
    plot.set(ylim=(0,1))
    plot.set(xlim=(0, min(len(frames_new), len(frames_odd))))
    plot.set_xticklabels(range(min(len(frames_new), len(frames_odd))))
    plot.set_xticklabels(size=15) 
    plot.set_yticklabels(size=15)
    plot.set_xlabels("Iteration", fontsize=20)
    plot.set_ylabels("Activation", fontsize=20)
    
    
    
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
    
    
    palette = sns.color_palette("plasma")
    sns.set_theme(style="ticks")
    plot = sns.relplot(data = df, x='Iteration', y='Activation', col="Slot", style="Word", kind="line", hue="Word", linewidth=8)
    plot.set(ylim=(0,1))
    plot.set(xlim=(0, min(len(frames_new), len(frames_odd))))
    plot.set_xticklabels(range(min(len(frames_new), len(frames_odd))))
    plot.set_xticklabels(size=15) 
    plot.set_yticklabels(size=15)
    plot.set_xlabels("Iteration", fontsize=20)
    plot.set_ylabels("Activation", fontsize=20)



    plt.show()
