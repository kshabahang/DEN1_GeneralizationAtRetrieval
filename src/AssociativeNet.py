import sys
import pickle
from itertools import combinations, product
#import cv2
import numpy as np
from scipy.linalg import hadamard, norm
from numpy.linalg import eig, eigh
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
            if self.hparams["sparse"]:
                self.feedback = self.feedback_stp_sparse
            else:
                self.feedback = self.feedback_stp_dense
        self.E = [] #environment vectors for dealing with {-1,1} vectors



    def report(self, X_order, verbose=True):
        '''probe system and return the most active item for idx_predict'''
        self.echo_full = X_order
        self.feedback()
        idx_predict = self.idx_predict
        i_start = len(self.vocab)*idx_predict
        i_end   = len(self.vocab)*(idx_predict+1)
        report = self.vocab[np.argmax([ self.strengths[i_start:i_end]])]
        return report

    def __sub__(self, other):
        self.lesion(other)

    def lesion(self, pair):
        w1, w2 = pair
        self.lesion_pair = pair
        K = self.K
        V = len(self.vocab)
        D = len(self.Ds)

        a2b = deepcopy(self.W[self.I[w1],V + self.I[w2]])
        b2a = deepcopy(self.W[V + self.I[w2],self.I[w1]])
        self.W[self.I[w1],V + self.I[w2]] = 0
        self.W[V + self.I[w2],self.I[w1]] = 0
        self.a2b = a2b
        self.b2a = b2a

    def __invert__(self):
        self.reverse_lesion()

    def reverse_lesion(self):
        w1, w2 = self.lesion_pair
        K = self.K
        V = len(self.vocab)
        D = len(self.Ds)

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
                    #bank_k.append(self.echo_full[k*self.N + i])
                    #sts = 1*self.echo_full[k*self.N + i]
                    #sts = sts.clip(min=0)
                    #bank_k.append(sts)
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
        if self.hparams['explicit'] and not self.hparams['init_weights']:
            '''In this case we also construct an explicit matrix'''
           
            if self.hparams['sparse']:
                self.W = lil_matrix(self.COUNTS.shape, dtype=float128)#np.zeros(self.COUNTS.shape)#
                for p in range(K): #TODO speed this up by exploiting the symmetry
                   for q in range(p, K):
                       C_pq = self.COUNTS[p*V:(p+1)*V, q*V:(q+1)*V]
                       #W_pq = self.Ds[0][p*V:(p+1)*V, q*V:(q+1)*V]
                       D_pq = self.Ds[0][p*V:(p+1)*V, q*V:(q+1)*V].tolil()
                       print(C_pq.shape)
                       C_pq.eliminate_zeros()
                       if binaryMat:
                           for i in range(len(C_pq.data)):
                               if C_pq.data[i] > 2:
                                   C_pq.data[i] = 1
                               else:
                                   C_pq.data[i] = 0
                           C_pq.eliminate_zeros()
                           C_pq = W_pq.tolil()
                           for i in range(V):
                               for j in range(len(C_pq.rows[i])):
                                   self.W[p*V + i, q*V + W_pq.rows[i][j]] = C_pq.data[i][j]#/len(W_pq.rows[i])
                                   self.W[q*V + W_pq.rows[i][j], p*V + i] = C_pq.data[i][j]#/len(W_pq.rows[i])
                       else:
                           alpha = 1 #add-1 Laplacian
                           logk = np.log2(7)


                           #experimental: this is for exploring new normalizations
                           #nnzsi = C_pq.getnnz(axis=1)
                           #nnzsj = C_pq.getnnz(axis=0)
                           print(C_pq.nnz)


                           #for i in range(len(C_pq.data)):
                           #    if C_pq.data[i] > 2:
                           #        C_pq.data[i] = 1
                           #    else:
                           #        C_pq.data[i] = 0                              
                           #C_pq.eliminate_zeros() 

                           T = C_pq.sum()
                           Pi = (np.array(C_pq.sum(axis=1).T)[0] + alpha*V)/(T + alpha*(V**2))
                           Pj = (np.array(C_pq.sum(axis=0))[0] + alpha*V)/(T + alpha*(V**2))
                           P_pq = (C_pq.todense() + alpha)/(T + alpha*(V**2))
                           weights = np.log2(np.diag(1.0/Pi).dot(P_pq.dot(np.diag(1.0/Pj)))) - logk#/-np.log2(P_pq)
                           weights.clip(min=0, out=weights)

                           if p != q:
                               for i in range(V):
                                   weights -= np.eye(V)

                           weights=  csr_matrix(weights)

                           self.W[p*V:(p+1)*V, q*V:(q+1)*V] = weights
                           self.W[q*V:(q+1)*V, p*V:(p+1)*V] = weights.T #global symmetry



                           #W_pq = W_pq.tolil()
                           

                           #S = W_pq.sum()

                           #sumi=(np.array(W_pq.sum(axis=1).T)[0]+1)/S
                           #sumj=(np.array(W_pq.sum(axis=0))[0]+1)/S
                           #if p != q:
                           #if True:
                           # for i in range(V):
                           #     for j in range(len(W_pq.rows[i])):
                           #         self.W[p*V+i,q*V+W_pq.rows[i][j]] = 1.0/nnzsj[W_pq.rows[i][j]]
                           #         self.W[q*V+W_pq.rows[i][j], p*V+i] = 1.0/nnzsi[i]
                           #     #for j in range(len(D_pq.rows[i])):
                           #     #    self.W[p*V+i,q*V+D_pq.rows[i][j]] = 1 
                           #     #    self.W[q*V+D_pq.rows[i][j], p*V+i] = 1


                self.W = self.W.tocsr()
                #self.update_eig()
                self.W.eliminate_zeros() 
            else: #dense
                self.W = np.zeros(self.COUNTS.shape)
                alpha = 0.01 #add-1 Laplacian
                #logk = np.log2(7)
                for p in range(K): 
                   for q in range(p, K): #we can exploit the symmetry here
                       print(p, q)
                       C_pq = self.COUNTS[p*V:(p+1)*V, q*V:(q+1)*V]

                       #nnzsi = C_pq.getnnz(axis=1)
                       #nnzsj = C_pq.getnnz(axis=0)
                       T = C_pq.sum()
                       Pi = (np.array(C_pq.sum(axis=1).T)[0] + alpha*V)/(T + alpha*(V**2))
                       Pj = (np.array(C_pq.sum(axis=0))[0] + alpha*V)/(T + alpha*(V**2))
                       P_pq = (C_pq.todense() + alpha)/(T + alpha*(V**2))
                       weights = np.log2(np.diag(1.0/Pi).dot(P_pq.dot(np.diag(1.0/Pj))))# - logk#/-np.log2(P_pq)
                       if p != q:
                           for i in range(V):
                               if weights[i,i] > 0:
                                weights[i,i] *= -1
                       print("T = {}".format(T))
                       #weights.clip(min=0, out=weights)
                       #weights = C_pq.todense() - np.outer(np.array(C_pq.sum(axis=0))[0]/T, C_pq.sum(axis=1))
                       #mu_j = weights.mean(axis=1)
                       #sig_j= weights.std(axis=1)
                       #weights = (weights - mu_j)/sig_j
                       #for i in range(V):
                       #    weights[i][i] = 0
                       #mu_i = weights.mean(axis=1)
                       #sig_i = weights.std(axis=1)
                       #weights = ((weights.T - mu_i)/sig_i).T

                       #weights = np.diag(1.0/np.log2(nnzsi + 1e-7)).dot(weights.dot(np.diag(1.0/np.log2(nnzsj + 1e-7))))
                       #for i in range(V):
                       #    weights[i][i] = 0
                       self.W[p*V:(p+1)*V, q*V:(q+1)*V] = weights
                       self.W[q*V:(q+1)*V, p*V:(p+1)*V] = weights.T #global symmetry
                del weights
                #del P_pq
                del C_pq
                #self.update_eig()
                
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


    def prune(self, min_wf = 3):
        '''go through the co-occurrence matrix and get rid of every entry with word-freq less than min_wf'''
        V0 = self.V
        K = self.K
        counts_mask = np.zeros(V0*K)
        #vocab_mask = self.COUNTS[:V0, :V0].diagonal() >= min_wf
        vocab_mask = np.array(self.COUNTS[:self.V, self.V:].sum(axis=0))[0] >= min_wf
        for k in range(K):
            counts_mask[k*V0:(k+1)*V0] = vocab_mask
        counts_mask = counts_mask.astype(bool)
        self.COUNTS = self.COUNTS[counts_mask, :][:, counts_mask]


        vocab_new = np.array(self.vocab)[vocab_mask]
        V = len(vocab_new)
        #remap vocab and I
        I_new = {vocab_new[i]:i for i in range(V)}

        self.V = V
        self.vocab = vocab_new
        self.I = I_new
        print(V0, V)
        percent_kept =  round((V/V0)*100, 2)
        print("Pruned vocabulary by discarding terms with frequency lower than {}".format(min_wf))
        print("Vocabulary size reduced from {} to {} ({}%)".format(V0, V, percent_kept ))


    def update_eig(self):
        if self.hparams["sparse"]:
            ei, ev = eigs(self.W.astype(np.float64), k =50)
        else:
            ei, ev = eigh(self.W)
            ei = ei[::-1]
            ev = ev[:, ::-1]
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
            ev2 = ev[self.N:]
            e1_sign = ""
            e2_sign = ""
            #abs_min = np.abs(ev1.min())
            #abs_max = np.abs(ev1.max())
            #if abs_min > abs_max:
            #    ev1 = -ev1 
            #    e1_sign = " - "
            #else:
            #    e1_sign = " + "

            #abs_min = np.abs(ev2.min())
            #abs_max = np.abs(ev2.max())
            #if abs_min > abs_max:
            #    ev2 = -ev2 
            #    e2_sign = " - "
            #else:
            #    e2_sign = " + "


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

    def feedback_stp_dense(self):
        '''recurrence with stp (i.e., DEN)'''
        ###compute strengths for the initial input pattern in buffer
        #self.compute_sts()
        #self.sort_banks()
        self.top_weights = []
        #self.view_banks(5)
        frames=  [deepcopy(self.echo_full)]

        #self.get_top_connect() 
    

        vlens = [(norm(self.echo_full), norm(self.echo_full[:self.V]), norm(self.echo_full[self.V:]))]

        print("Adding STP")
        ###compute the next state
        x0 = 1*self.echo_full #for STP
        nnz0 = np.where(x0 != 0)[0]
        for ii in nnz0:
            for jj in nnz0:
                self.W[ii, jj] += self.alpha*x0[ii]*x0[jj]

        try: #this is here to ensure stp is taken back out in case of interruption..feel free to crt-c out
            x = 1*self.echo_full
            x_new = x.dot(self.W)
            vlen = float(norm(x_new))
            vlen1= float(norm(x_new[:self.V]))
            vlen2= float(norm(x_new[self.V:]))
            vlens.append((vlen, vlen1, vlen2))
            x_new = x_new/vlen
            diff = norm(x - x_new)
            count = 0

            while(diff > self.eps and count < self.maxiter):
                count += 1
                ###load buffer with new state
                x = 1*x_new
                self.echo_full = 1*x_new #1*x_new

                #self.get_top_connect()
                self.compute_sts() #compute strengths with updated buffer
                self.sort_banks() 
                self.view_banks(5)


                ###compute the next state
                x_new = x.dot(self.W) #took out implicit for now

                vlen = float(norm(x_new))
                vlen1= float(norm(x_new[:self.V]))
                vlen2= float(norm(x_new[self.V:]))
                vlens.append((vlen, vlen1, vlen2))

                frames.append(deepcopy(x_new ))
                x_new = x_new/vlen

                diff =  float(norm(x - x_new))


            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= self.alpha*x0[ii]*x0[jj]
                
        except Exception as e:
            count = 0 #reset weights to before stp
            print(e)
            for ii in nnz0:
                for jj in nnz0:
                    self.W[ii, jj] -= self.alpha*x0[ii]*x0[jj]





        self.echo_full = 1*x_new

        print(self.echo_full.shape)

        self.compute_sts() #compute strengths with updated buffer
        self.sort_banks()
        self.view_banks(5)
        self.frames = frames
        self.vlens  = vlens
        self.count = count



    def feedback_stp_sparse(self):
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
                self.echo_full = np.array(x_new.todense())[0]#1*x_new

                #self.get_top_connect()
                self.compute_sts() #compute strengths with updated buffer
                self.sort_banks() 
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
                #if bank_data[1][j]  >= 0.00000000001:
                if True:
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
















































































































































































































































