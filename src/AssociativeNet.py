import sys

from itertools import combinations, product
#import cv2
import numpy as np
from scipy.linalg import hadamard, norm

from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix, csr_matrix
from copy import deepcopy
from GeneralNetModel import Model
from progressbar import ProgressBar

class AssociativeNet(Model):
    def __init__(self, hparams):
        super(AssociativeNet, self).__init__(hparams )
        self.N = hparams["N"] # dimensionality
        self.K = hparams["numBanks"] # num banks
        self.idx_predict = hparams["idx_predict"]
        self.echo_full = np.zeros(self.K*self.N) #state vector
        self.C = hparams["C"]
        self.COUNTS = None
        self.eps = hparams["eps"]
        self.alpha = hparams["alpha"]
        self.beta  = hparams["beta"]
        self.attractors = []
        self.vocab = []
        self.eta = hparams["eta"]

        self.vlens = [] #used to store vector lengths during recurrence
        self.energy= [] #Lyapunov energy

        if hparams["localist"]:
            self.MatMul = self.ImplicitMatMul_localist
        elif hparams["distributed"]:
            self.MatMul = self.ImplicitMatMul_distributed
        if hparams["explicit"]:
            self.MatMul = self.ExplicitMatMul
            if hparams['init_weights']:
                self.W = np.zeros((self.K*self.N, self.K*self.N)) #will need the explicit matrix

        if hparams["feedback"] == "linear":
            print ("Linear associator")
            self.feedback = self.feedback_lin
        elif hparams["feedback"] == "saturate":
            print ("Brain-State-in-a-Box")
            self.feedback = self.feedback_sat
        elif hparams["feedback"] == "stp":
            print ("Dynamic Eigen Net")
            self.feedback = self.feedback_stp
        self.E = [] #environment vectors for dealing with {-1,1} vectors


    def saturate(self, x):
        C = self.C

        for i in range(self.N*self.K):
            if x[i] > C:
                x[i] = C
            if x[i] < -C:
                x[i] = -C
        return x

    def report(self, X_order, verbose=True):
        self.echo_full = X_order
        self.feedback()
        idx_predict = self.idx_predict
        report = self.vocab[np.argmax([ self.strengths[len(self.vocab)*idx_predict:len(self.vocab)*(idx_predict+1)]  ])]
        return report


    def compute_sts(self):
        banks = []
        for k in range(self.K):
            bank_k = []
            if self.hparams["localist"]:
                for i in range(len(self.vocab)):
                    bank_k.append(np.abs(self.echo_full[k*self.N + i]))
                banks += bank_k
            elif self.hparams["distributed"]:
                for i in range(len(self.vocab)):
                    #bank_k.append( np.abs(vcos(np.array(self.E[i].todense())[0], self.echo_full[k*self.N:(k+1)*self.N]))) 
                    bank_k.append( np.exp(vcos(np.array(self.E[i]), self.echo_full[k*self.N:(k+1)*self.N]))) 
                #bank_k = list(np.array(bank_k)/sum(bank_k))
                banks += bank_k



        self.strengths = np.array(banks)

    def compute_weights(self, binaryMat=True):
        K = self.K 
        V = len(self.vocab)
        N = self.N

        if self.hparams['explicit'] and not self.hparams['init_weights']:
            if self.hparams['localist']:
                self.W = np.zeros(self.COUNTS.shape)#
                #self.W = lil_matrix(self.COUNTS.shape)

                for p in range(K):
                    for q in range(K):
                        W_pq = self.COUNTS[p*V:(p+1)*V, q*V:(q+1)*V]
                            
#                        binaryMat = True
                        if binaryMat:

                            for i in range(len(W_pq.data)):
                                if W_pq.data[i] > 2:
                                    W_pq.data[i] = 1
                                else:
                                    W_pq.data[i] = 0
                            W_pq.eliminate_zeros()
                            print(W_pq.nnz)
#                            W_pq.data = W_pq.data/650.0
                            W_pq = W_pq.tolil()
                            for i in range(V):
                                for j in range(len(W_pq.rows[i])):
                                    self.W[p*V + i, q*V + W_pq.rows[i][j]] = W_pq.data[i][j]
                        else:
                            #experimental
                            nnzs = W_pq.getnnz(axis=1)
                            #for i in range(len(W_pq.data)):
                            #    if W_pq.data[i] > 2:
                            #        W_pq.data[i] = 1
                            #    else:
                            #        W_pq.data[i] = 0
                            #W_pq.eliminate_zeros()
                            print(W_pq.nnz)
                            #W_pq.data = W_pq.data/650.0
                            W_pq = W_pq.tolil()
                            for i in range(V):
                                for j in range(len(W_pq.rows[i])):
                                    self.W[p*V + i, q*V + W_pq.rows[i][j]] = float(  np.log2(1 + W_pq[i,j]) /(nnzs[i]*nnzs[j])  )
            else: #self.hparams['distributed']
                #self.W = np.zeros(self.COUNTS.shape)#
                self.W = lil_matrix(self.COUNTS.shape) #can't use sparse because W will be dense                                                                               
                for p in range(K):
                    for q in range(K):
                        W_pq = self.COUNTS[p*V:(p+1)*V, q*V:(q+1)*V]
                            
                        #binaryMat = True
                        if binaryMat:
                                                                                                                                  
                            for i in range(len(W_pq.data)):
                                if W_pq.data[i] > 2:
                                    W_pq.data[i] = 1
                                else:
                                    W_pq.data[i] = 0
                            W_pq.eliminate_zeros()
                            print(W_pq.nnz)
                             #W_pq.data = W_pq.data/650.0
                            W_pq = W_pq.tolil()
                            for i in range(V):
                                pbar = ProgressBar(maxval = len(W_pq.rows[i])).start()
                                for j in range(len(W_pq.rows[i])):
                                    #self.W[p*V + i, q*V + W_pq.rows[i][j]] = 
                                    self.W[p*N:(p+1)*N, q*N:(q+1)*N] += W_pq.data[i][j]*np.outer(self.E[i], self.E[ W_pq.rows[i][j] ])
                                    print(W_pq.rows[i][j])
                                    pbar.update(j+1)
                        else:
                            #experimental
                            nnzs = W_pq.getnnz(axis=1)
                            print(W_pq.nnz)
                            #W_pq.data = W_pq.data/650.0
                            W_pq = W_pq.tolil()
                            for i in range(V):
                                for j in range(len(W_pq.rows[i])):
                                    self.W[p*V + i, q*V + W_pq.rows[i][j]] = float(  np.log2(1 + W_pq[i,j]) /(nnzs[i]*nnzs[j])  )

            self.W = self.W.tocsr()
            #if self.hparams["feedback"] == "stp":
            ei, ev = eigs(self.W, k =10)
            self.ei = ei.real
            self.ev = ev.real
            self.W.data /= self.ei[0] + 0.1

        else:
            self.WEIGHTS     = []
            self.WEIGHTS_IDX = []
            for p in range(K):
                weights_p = []
                weights_p_idx = []
                for q in range(K):
                    W_pq = self.COUNTS[p*V:(p+1)*V, q*V:(q+1)*V]
                    for i in range(len(W_pq.data)):
                        if W_pq.data[i] > 2:
                            W_pq.data[i] = 1
                        else:
                            W_pq.data[i] = 0
                    W_pq.eliminate_zeros()
                    W_pq.data = W_pq.data/300.0

                    weights_p.append(W_pq)
                    weights_p_idx.append(lil_matrix(W_pq).rows) #nonzero cell idxs for each row
                self.WEIGHTS_IDX.append(weights_p_idx)
                self.WEIGHTS.append(weights_p)

    def save_nonzero(self):
        self.E_nnz = lil_matrix(self.E).rows

    def ExplicitMatMul(self, X, X0):
        return X.dot(self.W) #+ np.outer(X0, X0))

    def ExplicitMatMulSparse(self, X, X0):
        return X.dot(self.W + X0.T.dot(X0) )


    def ImplicitMatMul_distributed(self, X, X0):
        '''Implicitly computes the matrix multiplication of a probe, X (size N)
           Uses the counts, C in (VK, VK), initial probe state, X0, and the environmental
           vectors, E in (V, N)'''
#        print("Matmul")
        N = self.N
        K = self.K
        V = len(self.vocab)
        Y = np.zeros(N*K)
        beta = self.beta
        for p in range(K):
            for q in range(K):
#                pbar= ProgressBar(maxval=V).start()
                for k in range(V):
                    for l in self.WEIGHTS_IDX[p][q][k]: #WEIGHT_IDX has the indices of nonzero cells for each row
                        dY = beta*self.WEIGHTS[p][q][k, l]*np.dot(X[q*N:(q+1)*N], self.E[l])*self.E[k]
                        Y[p*N:(p+1)*N] += dY
#                    pbar.update(k+1)
                        #Y[p*N:(p+1)*N] += dY#beta*self.WEIGHTS[p][q][k, l]*np.dot(X[q*N:(q+1)*N], self.E[l])*self.E[k] #Memory term #dense
#                        Y[p*N:(p+1)*N] += beta*self.WEIGHTS[p][q][k, l]*X[q*N:(q+1)*N].dot(self.E[l])*self.E[k] #sparse

                #Y[p*N:(p+1)*N] += np.dot(X0[q*N:(q+1)*N], X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term
                #Y[p*N:(p+1)*N] += X0[q*N:(q+1)*N].dot(X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term
        return Y

    def ImplicitMatMul_distributed_sp(self, X, X0):
        '''Implicitly computes the matrix multiplication of a probe, X (size N)
           Uses the counts, C in (VK, VK), initial probe state, X0, and the environmental
           vectors, E in (V, N)'''
        print("Matmul")
        N = self.N
        K = self.K
        V = len(self.vocab)
        Y = lil_matrix((1, N*K))
        beta = self.beta
        for p in range(K):
            for q in range(K):
                pbar = ProgressBar(maxval = V).start()
                for k in range(V):
                    for l in self.WEIGHTS_IDX[p][q][k]: #WEIGHT_IDX has the indices of nonzero cells for each row
                        Y[0, p*N:(p+1)*N] += beta*self.WEIGHTS[p][q][k, l]*X[0, q*N:(q+1)*N].dot(self.E[l].T)*self.E[k] #sparse
                    pbar.update(k+1)

                #Y[p*N:(p+1)*N] += np.dot(X0[q*N:(q+1)*N], X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term
                #Y[0, p*N:(p+1)*N] += X0[q*N:(q+1)*N].dot(X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term

        return Y




    def ImplicitMatMul_localist(self, X, X0):
        '''Implicitly computes the mat N = self.Nrix multiplication of a probe, X (size N)N
           Uses the counts, C in (VK, VK), initial probe state, X0, and the environmental
           vectors, E in (V, N)''' 
        N = self.N
        K = self.K
        V = len(self.vocab)
        Y = np.zeros(self.N*self.K)
        beta = self.beta
        for p in range(K):
            for q in range(K):
                for k in range(V):
                    for l in self.WEIGHTS_IDX[p][q][k]: #WEIGHT_IDX has the indices of nonzero cells for each row
                        Y[p*N + k] += beta*self.WEIGHTS[p][q][k, l]*X[q*N + l] #Memory term
                Y[p*N:(p+1)*N] += np.dot(X0[q*N:(q+1)*N], X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term
        return Y

#    def feedback_sat(self):
#        
#        ###compute strengths for the initial pattern in buffer
#        self.compute_sts()
#        self.sort_banks()
#        frames=  [deepcopy(self.strengths)]
#        echo_frames = [deepcopy(self.echo_full)]
#
#        ###compute the next state
#        x_new = self.MatMul(self.echo_full, 0*self.echo_full) #TODO: find a better way to deal with STP term 
#        x_new = self.saturate(x_new)
#        diff = norm(self.echo_full - x_new)
#
#        while(diff > self.eps):
#            ###load buffer with new state
#            self.echo_full = 1*x_new
#            self.compute_sts() #compute strengths with updated buffer
#            self.sort_banks()
#            frames.append(deepcopy(self.strengths))
#            echo_frames.append(deepcopy(self.echo_full))
#
#            ###compute the next state
#            x_new = self.MatMul(self.echo_full, 0*self.echo_full)
#            x_new = self.saturate(x_new)
#            diff = norm(self.echo_full - x_new)
#
#        self.echo_full = 1*x_new
#        self.compute_sts() #compute strengths with updated buffer
#        self.sort_banks()
#        frames.append(deepcopy(self.strengths))
#        echo_frames.append(deepcopy(self.echo_full))
#
#
#        self.frames = frames
#        self.echo_frames = echo_frames
    def feedback_sat(self):
        from scipy.sparse.linalg import norm as norm_sp
        ###compute strengths for the initial input pattern in buffer
        self.compute_sts()
        self.sort_banks()

        self.view_banks(5)
        frames=  [deepcopy(self.strengths)]
        echo_frames = [deepcopy(self.echo_full)]

        vlens = [np.linalg.norm(self.echo_full)]
        energy = []
#        energy= [-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full)]

        ###compute the next state
        #x0 = self.alpha*self.echo_full #for STP
        #self.W += np.outer(x0, x0)
        x0 = self.echo_full#csr_matrix(self.echo_full)
        x = self.echo_full#csr_matrix(self.echo_full)
        x_new = self.MatMul(x, 0*x0)
        vlen = float(norm(x_new)) 
        vlens.append(vlen)
        #x_new = x_new/vlen
        x_new.clip(min=-1, max=1, out=x_new)
        diff = float(norm(x0 - x_new))
        count = 0

        while(diff > self.eps):
            count += 1
            ###load buffer with new state
            x = 1*x_new
            self.echo_full = 1*x_new
            self.compute_sts() #compute strengths with updated buffer
            self.sort_banks()
            self.view_banks(5)
#            frames.append(deepcopy(self.strengths))
#            echo_frames.append(deepcopy(self.echo_full))




            ###compute the next state
            x_new = self.MatMul(x, 0*x)#self.echo_full.dot(self.W)#self.MatMul(self.echo_full, x0) 
            vlen = float(norm(x_new))
            vlens.append(vlen)

            #x_new = x_new/vlen
            x_new.clip(min=-1, max=1, out=x_new)
#            energy.append(-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full))
            diff =  float(norm(x - x_new))
            
            print(diff, count)

 #           print "-"*32
 #           self.view_banks(5)
 #           print count, diff

        #this way we always reset memory if we have to exit
        #self.W -= np.outer(x0, x0)

        self.echo_full = 1*x_new#1*np.array(x_new.todense())[0]
        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
#        frames.append(deepcopy(self.strengths))
#        echo_frames.append(deepcopy(self.echo_full))
#        energy.append(-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full))


        self.frames = frames
        self.echo_frames = echo_frames
        self.vlens  = vlens
        self.count = count


    def feedback_lin(self):

        ###compute strengths for the initial input pattern in buffer
        self.compute_sts()
        self.sort_banks()
        frames=  [deepcopy(self.strengths)]
        echo_frames = [deepcopy(self.echo_full)]

        ###compute the next state
        x_new = self.MatMul(self.echo_full, 0*self.echo_full) 
        x_new = x_new/norm(x_new)
        diff = norm(self.echo_full - x_new)

        while(diff > self.eps):
            ###load buffer with new state
            self.echo_full = 1*x_new
            self.compute_sts() #compute strengths with updated buffer
            self.sort_banks()
            frames.append(deepcopy(self.strengths))
            echo_frames.append(deepcopy(self.echo_full))

            ###compute the next state

            x_new = self.MatMul(self.echo_full, 0*self.echo_full)
            x_new = x_new/norm(x_new)
            diff = norm(self.echo_full - x_new)

        self.echo_full = 1*x_new
        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
        frames.append(deepcopy(self.strengths))
        echo_frames.append(deepcopy(self.echo_full))


        self.frames = frames
        self.echo_frames = echo_frames

    def feedback_stp(self):
        from scipy.sparse.linalg import norm as norm_sp
        ###compute strengths for the initial input pattern in buffer
        self.compute_sts()
        self.sort_banks()

        self.view_banks(5)
        frames=  [deepcopy(self.strengths)]
        echo_frames = [deepcopy(self.echo_full)]

        vlens = [np.linalg.norm(self.echo_full)]
        energy = []
#        energy= [-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full)]

        print("Adding STP")
        ###compute the next state
        x0 = self.alpha*self.echo_full #for STP
        nnz0 = np.where(x0 != 0)[0]
        for ii in nnz0:
            for jj in nnz0:
                self.W[ii, jj] += x0[ii]*x0[jj]
        #self.W += np.outer(x0, x0)
        try:
            x = csr_matrix(self.echo_full)
            x_new = x.dot(self.W)#self.MatMul(self.echo_full, x0)
            vlen = float(norm_sp(x_new)) 
            vlens.append(vlen)
            x_new = x_new/vlen
            diff = float(norm_sp(x - x_new))
            count = 0

            while(diff > self.eps):
                count += 1
                ###load buffer with new state
                x = 1*x_new
                #self.compute_sts() #compute strengths with updated buffer
                #self.sort_banks()
#                frames.append(deepcopy(self.strengths))
#                echo_frames.append(deepcopy(self.echo_full))




                ###compute the next state
                x_new = x.dot(self.W)#self.echo_full.dot(self.W)#self.MatMul(self.echo_full, x0) 
                vlen = float(norm_sp(x_new))
                vlens.append(vlen)

                x_new = x_new/vlen
#                energy.append(-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full))
                diff =  float(norm_sp(x - x_new))


            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= x0[ii]*x0[jj]
                
#                print(diff, count)

 #               print "-"*32
 #               self.view_banks(5)
 #               print count, diff
        except Exception as e:
            print(e)
            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= x0[ii]*x0[jj]
        #this way we always reset memory if we have to exit
        #self.W -= np.outer(x0, x0)

        self.echo_full = 1*np.array(x_new.todense())[0]
        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
#        frames.append(deepcopy(self.strengths))
#        echo_frames.append(deepcopy(self.echo_full))
#        energy.append(-0.5*self.echo_full.T.dot(self.W).dot(self.echo_full))


        self.frames = frames
        self.echo_frames = echo_frames
        self.vlens  = vlens
        self.count = count




    
    ###over-riding general model funcs to accomodate {-1, 1} vectors
    def update_vocab(self, corpus_process, wvecs = {}):
        for i in range(len(corpus_process)):
            if corpus_process[i] in self.wf and corpus_process[i] != "_":
                self.wf[corpus_process[i]] += 1
            elif corpus_process[i] != "_":
                ###construct new environment vector
                self.wf[corpus_process[i]] = 1
                self.I[corpus_process[i]] = len(self.vocab)
                self.vocab.append(corpus_process[i])
        #        if corpus_process[i] in wvecs:
        #            self.E.append(list(wvecs[corpus_process[i]]))
        #        else:
                if self.hparams["distributed"]:
                    e = np.hstack([np.ones(int(self.N/2)), -1*np.ones(int(self.N/2))]) #dense
#                    e = np.hstack([np.ones(int(10/2)), -1*np.ones(int(10/2)), np.zeros(self.N - 10)]) #spatter-codes 
                    np.random.shuffle(e)
                #elif self.hparams["localist"]:
                #    e = np.zeros(self.N)
                #    e[len(self.vocab)-1] = 1
                    self.E.append(list(e))

            self.V = len(self.vocab)



    def sent2vec(self, input_sentence):
        numSlots = self.hparams["numSlots"]
        iShiftS = max(len(input_sentence) - numSlots, 0)

        o = deepcopy(self.nullvec)
        if self.hparams["localist"]:
            for i in range(min(len(input_sentence), numSlots)):
                w_i = input_sentence[iShiftS + i]
                o[i*self.N+self.I[w_i]] += 1
        elif self.hparams["distributed"]:
            for i in range(min(len(input_sentence), numSlots)):
                w_i = input_sentence[iShiftS + i]
                o[i*self.N:(i+1)*self.N] = self.E[self.I[w_i]]
        return o
    def encode(self, sentence, st = 1.0, toNorm = True):
        X_o = self.sent2vec(sentence.split())
        if toNorm:
            X_o = X_o/norm(X_o)
        self.W += st*np.outer(X_o, X_o)

    def probe(self, sentence, st = 1.0, toNorm = True, noise = 0, verbose=False):
        X_order = self.sent2vec(sentence.split())
        X_order = st*X_order + np.random.normal(0, noise, self.N*self.K)
        if toNorm:
            X_order = X_order/norm(X_order)

        report = self.report(X_order)
        return report

    def sort_banks(self, echo = None):
        from numpy import argsort, array
        #sort all the banks in non-ascending order of activations
        numBanks = self.hparams['numBanks']
        bank_labels = self.hparams['bank_labels']
        echo = deepcopy(self.strengths)
        V = len(self.vocab)
        #print echo
        for i in range(numBanks):
            bank = echo[i*V:(i+1)*V]
            isort = argsort(bank)[::-1]
            self.banks[bank_labels[i]] = [array(self.vocab)[isort], bank[isort].astype(float)]


    def view_banks(self, k, mute = False):
        '''print the top k most active words across all banks'''
        numBanks = self.hparams['numBanks']
        bank_labels = self.hparams['bank_labels']
        to_print = []
        for i in range(numBanks):
            s = ''
            bank_data = self.banks[bank_labels[i]]

            #print bank_data
            for j in range(len(bank_data[0][:k])):
                if bank_data[1][j]  >= 0.00000000001:
                    s += '%10s %5f ' % (bank_data[0][j], bank_data[1][j])
            to_print.append(bank_labels[i] + ' ' +  s)
        if mute:
            return to_print
        else:
            print( '\n'.join(to_print))

def vcos(u,v):
    udotv = u.dot(v)
    if udotv != 0:
        return udotv/(np.linalg.norm(u)*np.linalg.norm(v))
    else:
        return 0

def vcos_sparse(u,v):

    udotv = float(u.dot(v.T).todense())
    if udotv != 0:
        return udotv/(norm(u)*norm(v))
    else:
        return 0.0

if __name__ == "__main__":
    ANet = AssociativeNet({"V0":4, 
                            "N":1024, 
                            "numBanks":2, 
                            "C":1, 
                            "eps":0.01, 
                            "eta":0.01, 
                            "idx_predict":1, 
                            "alpha":1, 
                            "feedback":"linear", 
                            "numSlots":2, 
                            "beta":1})
















































































































































































































































