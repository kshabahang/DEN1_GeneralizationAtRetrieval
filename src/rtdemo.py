import sys, os
from AssociativeNet import *
#from matplotlib import pyplot as plt
#plt.ion()

#from plot_tools import *

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


def saturate(x):
    for i in range(len(x)):
        if x[i] < -0.5:
            x[i] = -0.5
        elif x[i] > 0.5:
            x[i] = 0.5
    return x


if __name__ == "__main__":

    N  = 784#64**2
    V0 = N
    K = 2

    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":0.001, 
               "eta":1,
               "alpha":1,
               "beta":1,
               "V0":V0,
               "N":N,
               "localist":False,
               "distributed":True,
               "explicit":True,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"stp",
               "gpu":False}
   #ANet = AssociativeNet(hparams)
   ###number of aestrisks (Ratcliff, Van Zandt, McKoon)
    N_trials = 60000
    Mu_low  = 38
    Mu_high = 56
    Sig = 14.4
    low  = np.random.normal(Mu_low, Sig, N_trials).astype(int).clip(15,100)
    high = np.random.normal(Mu_high, Sig, N_trials).astype(int).clip(15,100)
    stims_low =[]
    stims_high=[]
    low_resp = -np.ones(10)
    high_resp = np.ones(10)
    for i in range(N_trials):
        n_low = low[i]
        n_high= high[i]
        v_low = np.zeros(100)
        v_high = np.zeros(100)
        v_low[n_low-15:n_low+15] += 1
        v_high[n_high-15:n_high+15] += 1
        stims_low.append(v_low)
        stims_high.append(v_high)
    
    N = len(stims_low[0]) + len(low_resp)
    FB = "stp"

    ##params
    if FB == "stp":
        gamma = 1
        alpha = 0.2
        delta = 2
        eta   = 2
        decay = 1#0.995
        eps=0.05#49
    elif FB == "bsb":
        gamma = 1
        alpha = 0.2
        delta = 1
        eta   = 2
        decay = 0.995
        eps=0.01


    A = np.zeros((N, N))
    accuracy = []
    answers  = []
    true_ns = []
    cycles_raw  = []
    true_cond = []
    lat_resp0 = []
    lat_resp1 = []


    recurrent_fb = False
    for i in range(N_trials):
        cond = np.random.randint(2)
        true_cond.append(cond)
        ###probe system to get response
        if cond == 0:
            p = np.hstack([stims_low[i], np.zeros(10)])
            f = np.hstack([stims_low[i], low_resp])
            n_astrisks = low[i]
            true_ns.append(low[i])
        else:
            p = np.hstack([stims_high[i], np.zeros(10)])
            f = np.hstack([stims_high[i], high_resp])
            n_asterisks = high[i]
            true_ns.append(high[i])

        x0 = np.zeros(N) + p
        xt1 = x0*1
        if FB == "bsb":
            xt2 = saturate(gamma*xt1 + alpha*xt1.dot(A) + delta*x0)
        elif FB == "stp":
            #p = p/np.linalg.norm(p)
            A_prime = A + delta*np.outer(p, p)
            xt2 = alpha*xt1.dot(A_prime) 
            xt2 = xt2/np.linalg.norm(xt2)
        t = 1
        while(np.linalg.norm(xt1 - xt2) > eps):
            xt1 = xt2*1
            if FB == "bsb":
                xt2 = saturate(gamma*xt1 + alpha*xt1.dot(A) + delta*x0)
            elif FB == "stp":
                xt2 = alpha*xt1.dot(A_prime)
                xt2 = xt2/np.linalg.norm(xt2)
            t += 1

        resp = xt2[100:]
        cycles_raw.append(t)       
        st_low  = resp.dot(low_resp)
        st_high =resp.dot(high_resp)

#        st_low = np.exp(st_low)/(np.exp(st_high) + np.exp(st_low))
#        st_high =np.exp(st_high)/(np.exp(st_high) + np.exp(st_low))

        if st_low > st_high:
            ans = 0
        elif st_high > st_low:
            ans = 1
        else:
            ans = 0
        lat_resp0.append([t, ans == 0])
        lat_resp1.append([t, ans == 1])
        answers.append(ans)
        accuracy.append(ans == cond)


    
        ###feedback with full pattern
        if recurrent_fb:
            A_prime = A + np.outer(f,f)
            xt1 = 1*f
            xt2 = xt1.dot(A_prime)
            xt2 = xt2/np.linalg.norm(xt2)
            while(np.linalg.norm(xt1 - xt2) > 0.01):
                xt1 = xt2*1
                xt2 = xt1.dot(A_prime)
                xt2 = xt2/np.linalg.norm(xt2)
            A = A + eta*np.outer(xt2,xt2)
            f = f/np.linalg.norm(f)

        A = decay*A + eta*np.outer(resp[-1] - f,resp[-1] - f)



    error_idx = [i for i in range(N_trials) if (true_ns[i] < 47 and answers[i] == 1) or (true_ns[i] > 47 and answers[i] == 0)]
    error_idx = np.array(error_idx) 



    #fig, [f1,f2,f3] = plt.subplots(nrows=3, ncols=1)
    accuracy = np.array(accuracy).astype(int)
    pcorr = []
    k = 100
    for i in range(int(len(accuracy)/k)):
        pcorr.append(sum(accuracy[i*k:(i+1)*k])/float(k))
    #f1.plot(pcorr)

    byAsterisks = np.zeros((120, 2))
    byAsterisksRT = np.zeros((120, 2))
    byAsterisksN = np.zeros((120, 2))
    for i in range(len(answers)):
        byAsterisks[true_ns[i], answers[i]] += 1
        byAsterisksRT[true_ns[i], answers[i]] += cycles_raw[i]
        byAsterisksN[true_ns[i], answers[i]] += 1

    for i in range(len(byAsterisks)):
        if float(sum(byAsterisks[i])) != 0:
            byAsterisks[i] = byAsterisks[i]/float(sum(byAsterisks[i]))

#    fig = plt.figure()
#    f2.plot(filter(lambda x : x != 0, byAsterisks[15:-15][:,0])) #figure 27

    byAsterisksRT =  byAsterisksRT[15:-15]
    byAsterisksN  = byAsterisksN[15:-15] 


    low_curve = byAsterisksRT[:, 0]/(byAsterisksN[:, 0] + 0.0001)
    high_curve = byAsterisksRT[:, 1]/(byAsterisksN[:, 1] + 0.0001)

 #   fig = plt.figure()
#    f3.plot(filter(lambda x : x != 0, low_curve))
#    f3.plot(filter(lambda x : x != 0, high_curve))

    lat_resp0 = np.array(lat_resp0)
    lat_resp1 = np.array(lat_resp1)
    rt_bins0 = sorted(list(set(lat_resp0[:, 0])))
    cycles0  = np.zeros(len(rt_bins0))
    counts0  = np.zeros(len(rt_bins0))
    respbins0= np.zeros(len(rt_bins0))
    counts1 = np.zeros(len(rt_bins0))
    cycles1 = np.zeros((len(rt_bins0)))
    respbins1= np.zeros(len(rt_bins0))
    for i in range(len(lat_resp0)):
        idx = rt_bins0.index(lat_resp0[i][0])
#        cycles[idx] += lat_resp[i][0]

        if lat_resp0[i][1] == 1:
            counts0[idx] += 1
            respbins0[idx] += 1
            cycles0[idx]   += lat_resp0[i][0]
        else:
            counts1[idx] += 1
            respbins1[idx] += 1
            cycles1[idx] += lat_resp1[i][0]
#    respbins /= counts
#    cycles   /= counts

#    pcorr1 = respbins/counts
#    cycles1= cycles/counts
#    isort1 = np.argsort(pcorr1)
#    fig = plt.figure()
#    f4.plot(pcorr[isort], cycles[isort])

#    fig = plt.figure()
#    plt.hist([lat_resp[i][0] for i in range(len(lat_resp)) if lat_resp[i][1] == 1], label="Correct", alpha=0.5, normed=True)
#    plt.hist([lat_resp[i][0] for i in range(len(lat_resp)) if lat_resp[i][1] == 0], label="Incorrect", alpha=0.5, normed=True)
    
#    plt.legend()


    #rt_bins = sorted(list(set(lat_resp[:, 0])))
    #cycles  = np.zeros(len(rt_bins))
    #counts  = np.zeros(len(rt_bins))
    #respbins= np.zeros(len(rt_bins))
    #for i in range(len(lat_resp)):
    #    idx = rt_bins.index(lat_resp[i][0])
#   #     cycles[idx] += lat_resp[i][0]
    #    counts[idx] += 1
    #    if lat_resp[i][1] == 0:
    #        respbins[idx] += 1
    #        cycles[idx]   += lat_resp[i][0]
#   # respbins /= counts
#   # cycles   /= counts

    #pcorr2 = respbins/counts
    #cycles2= cycles/counts
    #isort2 = np.argsort(pcorr2)
    #pcorr_mean = 0.5*(pcorr1[isort1] + pcorr2[isort2])
    #cycles_mean = 0.5*(cycles1[isort1] + cycles2[isort2])

#    fig = plt.figure()
#    f4.plot(pcorr[isort], cycles[isort])
#    plt.plot(pcorr_mean, cycles_mean) 

    

    













































       
                












































































































































































































































































































































































































































































































































































































































































































































































