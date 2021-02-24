import sys
import pickle
from itertools import combinations, product
#import cv2
import numpy as np
from scipy.linalg import hadamard, norm

from scipy.sparse.linalg import eigs, eigen
from scipy.sparse import lil_matrix, csr_matrix
from scipy import float128
from copy import deepcopy
from GeneralNetModel import Model
from progressbar import ProgressBar

class AssociativeNet(Model):
    def __init__(self, hparams):
        super(AssociativeNet, self).__init__(hparams )
        self.N = hparams["N"] # dimensionality
        self.K = hparams["numBanks"] # num banks
        self.idx_predict = hparams["idx_predict"] #index of the bank we report from
        self.echo_full = np.zeros(self.K*self.N) #state vector
        self.COUNTS = None #co-occurrence matrix
        self.eps = hparams["eps"] #steady-state criterion
        self.alpha = hparams["alpha"] #short-term plasticity weight 
        self.beta  = hparams["beta"] 
        self.attractors = [] 
        self.vocab = [] #vocabulary
        self.eta = hparams["eta"]
        self.maxiter = hparams['maxiter']
        self.multi_degree = hparams['multiDegree']

        self.vlens = [] #used to store vector lengths during recurrence

        if hparams["explicit"]:
            self.MatMul = self.ExplicitMatMul
            if hparams['init_weights']:
                self.W = np.zeros((self.K*self.N, self.K*self.N)) #will need the explicit matrix
        if hparams["localist"]: #
            self.MatMul = self.ImplicitMatMul_localist
        elif hparams["distributed"]:
            self.MatMul = self.ImplicitMatMul_distributed

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



    def report(self, X_order, verbose=True):
        '''probe system and return the most active item for idx_predict'''
        self.echo_full = X_order
        self.feedback()
        idx_predict = self.idx_predict
        report = self.vocab[np.argmax([ self.strengths[len(self.vocab)*idx_predict:len(self.vocab)*(idx_predict+1)]  ])]
        return report

    def lesion(self, pair):
        w1, w2 = pair
        self.lesion_pair = pair
        K = self.K
        V = len(self.vocab)
        D = len(self.Ds)
        if self.multi_degree:
            a2bs = []
            b2as = []
            for l in range(D):
                i_shift = l*K*V
                a2b = deepcopy(ANet.W[i_shift + self.I[w1],i_shift + V + self.I[w2]])
                b2a = deepcopy(ANet.W[i_shift + V + self.I[w2], i_shift + self.I[w1]])
                a2bs.append(a2b)
                b2as.append(b2a)
                self.W[i_shift + self.I[w1],i_shift + V + self.I[w2]] = 0
                self.W[i_shift + self.N + self.I[w2],i_shift + self.I[w1]] = 0
            self.a2bs = a2bs
            self.b2as = b2as
        else:
            a2b = deepcopy(self.W[self.I[w1],V + self.I[w2]])
            b2a = deepcopy(self.W[V + self.I[w2],self.I[w1]])
            self.W[self.I[w1],V + self.I[w2]] = 0
            self.W[V + self.I[w2],self.I[w1]] = 0
            self.a2b = a2b
            self.b2a = b2a

    def reverse_lesion(self):
        w1, w2 = self.lesion_pair
        K = self.K
        V = len(self.vocab)
        D = len(self.Ds)
        if self.multi_degree:
            a2bs = self.a2bs
            b2as = self.b2as
            for l in range(D):
                i_shift = l*K*V
                a2b = deepcopy(a2bs[l])
                b2a = deepcopy(b2as[l])
                self.W[i_shift + self.I[w1],i_shift + V + self.I[w2]] = a2b
                self.W[i_shift + V + self.I[w2],i_shift + self.I[w1]] = b2a
        else:
            a2b = self.a2b
            b2a = self.b2a
            self.W[self.I[w1],V + self.I[w2]] = deepcopy(a2b)
            self.W[V + self.I[w2],self.I[w1]] = deepcopy(b2a)

    def compute_sts(self):
        '''compute strengths of activations from the echo'''

        banks = []
        for k in range(self.K):
            bank_k = []
            if self.hparams["localist"]:
                for i in range(len(self.vocab)):
                    bank_k.append(np.abs(self.echo_full[k*self.N + i]))
                banks += bank_k
            elif self.hparams["distributed"]:
                for i in range(len(self.vocab)):
                    bank_k.append( vcos(np.array(self.E[i]), self.echo_full[k*self.N:(k+1)*self.N])) 
                banks += bank_k

        self.strengths = np.array(banks)

    def compute_weights(self, binaryMat=True):
        '''Compute the weights from the co-occurrence counts'''
        K = self.K 
        V = len(self.vocab)
        self.V = V
        L = len(self.Ds)
        N = self.N
        if self.multi_degree:
            self.W = lil_matrix((K*V*L, K*V*L), dtype=float128)#np.zeros(self.COUNTS.shape)#
            print("Setting weights")
            pbar = ProgressBar(maxval=L).start()
            for l in range(L):
                for p in range(K): #
                   for q in range(p, K):
                       if l == 0:
                           D = self.Ds[l][p*V:(p+1)*V, q*V:(q+1)*V]
                       else:
                           D = self.Ds[l][p*V:(p+1)*V, q*V:(q+1)*V] - self.Ds[l-1][p*V:(p+1)*V, q*V:(q+1)*V]
                       W_pq = D.tolil()
                       #W_pq[W_pq > 0] = 0.7**l
                       shift_L = l*K*V
                       #self.W[shift_L + p*V:shift_L + p*V + V, shift_L + q*V :shift_L + q*V + V] = W_pq
                       #self.W[shift_L + q*V :shift_L + q*V + V ,shift_L + p*V:shift_L + p*V + V] = W_pq.T
                       if p == q:
                           for i in range(V):
                               self.W[shift_L + p*V + i,shift_L + q*V + i] =1# 0.7**l
                               self.W[shift_L + q*V + i,shift_L + p*V + i] =1# 0.7**l

                       for i in range(V):
                           for j in range(len(W_pq.rows[i])):
                               self.W[shift_L + p*V + i,shift_L + q*V + W_pq.rows[i][j]] = 0.8**l#W_pq.data[i][j]#/len(W_pq.rows[i])
                               self.W[shift_L + q*V + W_pq.rows[i][j],shift_L + p*V + i] = 0.8**l #W_pq.data[i][j]#/len(W_pq.rows[i])
                pbar.update(l+1)



            self.W = self.W.tocsr()
            self.update_eig()
            self.W.eliminate_zeros() 

        else:
            if self.hparams['explicit'] and not self.hparams['init_weights']:
                '''In this case we also construct an explicit matrix'''
                
                self.W = lil_matrix(self.COUNTS.shape, dtype=float128)#np.zeros(self.COUNTS.shape)#
                for p in range(K): #TODO speed this up by exploiting the symmetry
                   for q in range(p, K):
                       W_pq = self.Ds[0][p*V:(p+1)*V, q*V:(q+1)*V]
                       print(W_pq.shape)
                       W_pq.eliminate_zeros()
                       if binaryMat:
                           for i in range(len(W_pq.data)):
                               if W_pq.data[i] > 2:
                                   W_pq.data[i] = 1
                               else:
                                   W_pq.data[i] = 0
                           W_pq.eliminate_zeros()
                           W_pq = W_pq.tolil()
                           for i in range(V):
                               for j in range(len(W_pq.rows[i])):
                                   self.W[p*V + i, q*V + W_pq.rows[i][j]] = W_pq.data[i][j]#/len(W_pq.rows[i])
                                   self.W[q*V + W_pq.rows[i][j], p*V + i] = W_pq.data[i][j]#/len(W_pq.rows[i])
                       else:
                           #experimental: this is for exploring new normalizations
                           nnzsi = W_pq.getnnz(axis=1)
                           nnzsj = W_pq.getnnz(axis=0)
                           print(W_pq.nnz)


                           for i in range(len(W_pq.data)):
                               if W_pq.data[i] > 3:
                                   W_pq.data[i] = 1
                               else:
                                   W_pq.data[i] = 0                              
                           W_pq.eliminate_zeros()


                           W_pq = W_pq.tolil()
                           

                           S = W_pq.sum()

                           sumi=(np.array(W_pq.sum(axis=1).T)[0]+1)/S
                           sumj=(np.array(W_pq.sum(axis=0))[0]+1)/S
                           if p != q:
                            for i in range(V):
                                for j in range(len(W_pq.rows[i])):
                                    #self.W[p*V + i, q*V + W_pq.rows[i][j]] =  (W_pq[i,j]**2)/(sumi[i]*sumj[j])
                                    #pAB = (W_pq[i,j]+1)/S
                                    #pA = sumi[i]
                                    #pB = sumj[j]
                                    #pmi = np.log2((pAB)/(pA*pB))
                                    
                                    #strength = 1#W_pq[i,W_pq.rows[i][j]]#1#/len(W_pq.rows[i])#W_pq[i,W_pq.rows[i][j]]#np.max([np.log(W_pq[i,j]),0])
                                    #pmi = 2**(pmi + np.log2(pAB))
#                                    npmi = pmi/-np.log2(pAB)
                                    #ppmi = 2**(pmi + np.log2(pAB))
                                    #pmi = pAB/(pA*pB)
                                    #print(ppmi,pAB*(pAB/(pA*pB)) )#(pAB/(pA*pB))/2**(-np.log2(pAB)) )
                                    #print(pAB, pA, pB, pmi)
                                    #ppmi = np.max([0, pmi])
                                    self.W[p*V+i,q*V+W_pq.rows[i][j]] = 1.0/nnzsj[W_pq.rows[i][j]]
                                    self.W[q*V+W_pq.rows[i][j], p*V+i] = 1.0/nnzsi[i]

                self.W = self.W.tocsr()
                self.update_eig()
                self.W.eliminate_zeros() 

            else:
                '''No explicit matrix...matmul will be done implicitly'''
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




    def expand(self, x):
        D = len(self.Ds)
        return np.hstack([x]*D)

    def collapse(self, x, theta = None):
        D = len(self.Ds)
        K = self.K
        V = len(self.vocab)
        if theta == None:
            theta = np.array([1.0/D]*D)
        return np.array(theta.dot(x.reshape(D, K*V)))


    def update_eig(self):
        ei, ev = eigs(self.W.astype(np.float64), k =50)
        self.ei = ei.real
        self.ev = ev.real
        e_max = sorted(self.ei)[::-1][0]
        self.alpha = 1.001#e_max + 0.1*e_max
        self.W.data /= e_max #+ 0.1*e_max


    def print_eigenspectrum(self):
        for i in range(len(self.ei)):
            ei = self.ei[i]
            ev = self.ev[:, i]

            ev1 = ev[:self.N]
            ev2 = ev[:self.N]
            e1_sign = ""
            e2_sign = ""
            abs_min = np.abs(ev1.min())
            abs_max = np.abs(ev1.max())
            if abs_min > abs_max:
                ev1 = -ev1 #flip 'er
                e1_sign = " - "
            else:
                e1_sign = " + "

            abs_min = np.abs(ev2.min())
            abs_max = np.abs(ev2.max())
            if abs_min > abs_max:
                ev2 = -ev2 #flip 'er
                e2_sign = " - "
            else:
                e2_sign = " + "


            bank1 = sorted(zip(ev1, self.vocab))[::-1][:10]
            bank2 = sorted(zip(ev2, self.vocab))[::-1][:10]
            sts1, ws1 = zip(*bank1)
            sts2, ws2 = zip(*bank2)
            print(e1_sign, e2_sign, round(ei, 4), " ".join(ws1) + " | " + " ".join(ws2))



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
                Y[p*N:(p+1)*N] += X0[q*N:(q+1)*N].dot(X[q*N:(q+1)*N])*X0[p*N:(p+1)*N] #STP term
        return Y


    def ImplicitMatMul_localist(self, X, X0):
        '''Implicitly computes the mat N = self.Nrix multiplication of a probe, X (size N)N
           Uses the counts, C in (VK, VK), initial probe state, and X0''' 
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

    def feedback_sat(self):
        '''recurrence with saturation (cf. BSB)'''
        ###compute strengths for the initial input pattern in buffer
        self.compute_sts()
        self.sort_banks()

        self.view_banks(5)
        frames=  [deepcopy(self.strengths)]


        vlens = [norm(self.echo_full)]

        ###compute the next state
        x0 = 1*self.echo_full
        x = 1*self.echo_full
        x_new = self.MatMul(x, 0*x0)
        vlen = norm(x_new)
        vlens.append(vlen)

        x_new.clip(min=-1, max=1, out=x_new) #saturation
        diff = norm(x0 - x_new)
        count = 0
        while(diff > self.eps and count < self.maxiter):
            count += 1
            ###load buffer with new state
            x = 1*x_new
            self.echo_full = 1*x_new
            self.compute_sts() #compute strengths with updated buffer
            self.sort_banks()
            self.view_banks(5)
            frames.append(deepcopy(self.strengths))

            ###compute the next state
            x_new = self.MatMul(x, 0*x)#self.echo_full.dot(self.W)#self.MatMul(self.echo_full, x0) 
            vlen = float(norm(x_new))
            vlens.append(vlen)

            x_new.clip(min=-1, max=1, out=x_new) #saturation

            diff =  float(norm(x - x_new))
      
         

        self.echo_full = 1*x_new
        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
        frames.append(deepcopy(self.strengths))

        self.frames = frames
        self.vlens  = vlens
        self.count = count


    def feedback_lin(self):
        '''recurrence (linear)'''

        ###compute strengths for the initial input pattern in buffer
        self.compute_sts()
        self.sort_banks()
        frames=  [deepcopy(self.strengths)]


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


            ###compute the next state

            x_new = self.MatMul(self.echo_full, 0*self.echo_full)
            x_new = x_new/norm(x_new)
            diff = norm(self.echo_full - x_new)

        self.echo_full = 1*x_new
        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
        frames.append(deepcopy(self.strengths))

        self.frames = frames


    def feedback_stp(self):
        from scipy.sparse.linalg import norm as norm_sp
        '''recurrence with stp (i.e., DEN)'''
        ###compute strengths for the initial input pattern in buffer
        #self.compute_sts()
        #self.sort_banks()
        self.top_weights = []
        #self.view_banks(5)
        frames=  [deepcopy(self.echo_full)]

        self.get_top_connect() 
    

        vlens = [(norm(self.echo_full), norm(self.echo_full[:self.V]), norm(self.echo_full[self.V:]))]

        print("Adding STP")
        ###compute the next state
        x0 = 1*self.echo_full #for STP
        if self.multi_degree:
            x0 = self.expand(x0)
        nnz0 = np.where(x0 != 0)[0]
        for ii in nnz0:
            for jj in nnz0:
                self.W[ii, jj] += self.alpha*x0[ii]*x0[jj]

        try: #this is here to ensure stp is taken back out in case of interruption..feel free to crt-c out
            x = csr_matrix(1*x0, dtype=float128)#1*self.echo_full
            x_new = x.dot(self.W)
            vlen = float(norm_sp(x_new))
            vlen1= float(norm_sp(x_new[0, :self.V]))
            vlen2= float(norm_sp(x_new[0,self.V:]))
            vlens.append((vlen, vlen1, vlen2))
            x_new = x_new/vlen
            diff = norm_sp(x - x_new)
            count = 0

            while(diff > self.eps and count < self.maxiter):
                count += 1
                ###load buffer with new state
                x = 1*x_new
                if self.multi_degree:
                    self.echo_full = self.collapse(np.array(x_new.todense())[0])
                else:
                    self.echo_full = np.array(x_new.todense())[0]#1*x_new

                self.get_top_connect()
                self.compute_sts() #compute strengths with updated buffer
                self.sort_banks() #TODO decide how to save states during recurrence with multi-degree
                self.view_banks(5)


                ###compute the next state
                x_new = x.dot(self.W) #took out implicit for now

                vlen = float(norm_sp(x_new))
                vlen1= float(norm_sp(x_new[0, :self.V]))
                vlen2= float(norm_sp(x_new[0,self.V:]))
                vlens.append((vlen, vlen1, vlen2))

                frames.append(deepcopy(np.array(x_new.todense())[0]))
                x_new = x_new/vlen

                diff =  float(norm_sp(x - x_new))


            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= self.alpha*x0[ii]*x0[jj]
                
        except Exception as e:
            count = 0 #reset weights to before stp
            print(e)
            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= self.alpha*x0[ii]*x0[jj]


        if self.multi_degree:
            self.echo_full = self.collapse(np.array(x_new.todense())[0])
        else:
            self.echo_full = np.array(x_new.todense())[0]#1*x_new

        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
        self.view_banks(5)
        self.frames = frames
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
                if self.hparams["distributed"]:
                    e = np.hstack([np.ones(int(self.N/2)), -1*np.ones(int(self.N/2))]) #dense
#                    e = np.hstack([np.ones(int(10/2)), -1*np.ones(int(10/2)), np.zeros(self.N - 10)]) #spatter-codes 
                    np.random.shuffle(e)
                    self.E.append(list(e))

            self.V = len(self.vocab)

    def get_top_connect(self, top=20):
        V = self.V

        bank1 = np.argsort(self.echo_full[:V])[::-1][:top]
        bank2 = np.argsort(self.echo_full[V:])[::-1][:top]
        weights = np.zeros((top,top))
        ws1 =[]
        ws2 =[]
        sts1=[]
        sts2=[]
        for i in range(top):
            ws1.append(self.vocab[bank1[i]])
            ws2.append(self.vocab[bank2[i]])
            sts1.append(self.echo_full[bank1[i]])
            sts2.append(self.echo_full[V + bank2[i]])
            for j in range(top):
                weights[i,j] = self.W[bank1[i], V + bank2[j]]

        self.top_weights.append((ws1, ws2, sts1, sts2, weights, self.vlens))

    def save_top_weights(self, name_tag):
        f = open("top_weights{}.pkl".format(name_tag), "wb")
        pickle.dump(self.top_weights, f)
        f.close()



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
















































































































































































































































