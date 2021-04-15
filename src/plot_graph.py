from matplotlib import pyplot as plt
import networkx as nx
import pickle
import sys
import numpy as np

#corr_file = "top_weightsNNS_VBP_2_NN_VBP_children_learn_corr.pkl"
#incorr_file = "top_weightsNNS_VBP_2_NN_VBP_child_learn_incorr.pkl"
#
#cond = sys.argv[1]
#if cond == "corr":
#    f = open(corr_file, "rb")
#    top_weights = pickle.load(f)
#    f.close()
#else:
#    f = open(incorr_file, "rb")
#    top_weights = pickle.load(f)
#    f.close()
#n = int(sys.argv[2])
#k = int(sys.argv[3])
#(ws1, ws2, sts1, sts2, weights, vlens) = top_weights[n]
#
#vlens = vlens[n] #TODO fix this so we're not sending clones of vlen
#print("{} --- Vlen: {}  Vlen1: {}  Vlen2: {}".format(cond, round(vlens[0],4), round(vlens[1], 4), round(vlens[2], 4)))
#
#G = nx.Graph()
##G.add_nodes_from(ws1[:k] + ws2[:k])
#for i in range(k):
#    for j in range(k):
#        if weights[i,j] != 0:
#            G.add_edge( ws1[i], " "+ws2[j], weight=10*weights[i,j])
#
#edges = G.edges()
#weight= [G[u][v]['weight'] for u,v in edges]
#
#pos = nx.bipartite_layout(G, ws1) 
#fig = plt.figure()
#nx.draw(G, pos = pos, node_color="white", with_labels=True, alpha=0.5, width=weight)
#fig.show()

g1= np.load("../rsc/examples/g1.npy")
g2=np.load("../rsc/examples/g2.npy")
ug1=np.load("../rsc/examples/ug1.npy")
ug2=np.load("../rsc/examples/ug2.npy")
Wg = np.load("../rsc/examples/Wg.npy")
Wug=np.load("../rsc/examples/Wug.npy")

K = 10
weight_scale = 5000

G = nx.DiGraph()
for i in range(K):
    for j in range(i, K):
        if Wg[i, j] > 0:
            G.add_edge(g1[0][i], " " + g2[0][j], weight=weight_scale*Wg[i,j], color='red')
        elif Wg[i,j] == 0:
            G.add_edge(g1[0][i], " " + g2[0][j], weight=weight_scale*Wg[i,j], color='white')
        else:
            G.add_edge(g1[0][i], " " + g2[0][j], weight=weight_scale*Wg[i,j], color='blue')
        if Wg[j, i] > 0:
            G.add_edge(" " + g2[0][j], g1[0][i], weight=weight_scale*Wg[j,i], color='red')
        elif Wg[j,i] == 0:
            G.add_edge(" " + g2[0][j], g1[0][i], weight=weight_scale*Wg[j,i], color='white')
        else:
            G.add_edge(" " + g2[0][j], g1[0][i], weight=weight_scale*Wg[j,i], color='blue')
edges = G.edges()
weight= [G[u][v]['weight'] for u,v in edges]
color= [G[u][v]['color'] for u,v in edges]
#pos = nx.bipartite_layout(G, g1[0])


stmap = dict(zip(np.hstack([g1[0][:K],  [" "+g2[0][i] for i in range(K)]]), 
                 100*np.log(np.round(np.hstack([g1[1][:K], g2[1][:K]]).astype(float)*1e5) )   ))
stmap[g1[0][0]] /= 3
stmap[" " + g2[0][0]] /= 3



pos = {}
for i in range(K):
    pos[g1[0][i]] = [-1, 1 - i*0.1]
    pos[" " + g2[0][i]] = [1, 1 - i*0.1]


node_size = []
node_color= []
for node in G.nodes:
    node_size.append(stmap[node])
    node_color.append(stmap[node])


fig = plt.figure()
nx.draw(G, pos = pos,node_size=2500 , node_color=node_color, edge_color=color, with_labels=True, alpha=0.5, width=weight, cmap=plt.cm.bwr)

fig.show()


G = nx.DiGraph()
for i in range(K):
    for j in range(i, K):
        if Wug[i, j] > 0:
            G.add_edge(ug1[0][i], " " + ug2[0][j], weight=weight_scale*Wug[i,j], color='red')
        elif Wug[i,j] == 0:
            G.add_edge(ug1[0][i], " " + ug2[0][j], weight=weight_scale*Wug[i,j], color='white')
        else:
             G.add_edge(ug1[0][i], " " + ug2[0][j], weight=weight_scale*Wug[i,j], color='blue')
        if Wug[j, i] > 0:
            G.add_edge(" " + ug2[0][j], ug1[0][i], weight=weight_scale*Wug[j,i], color='red')
        elif Wug[j, i] == 0:
            G.add_edge(" " + ug2[0][j], ug1[0][i], weight=weight_scale*Wug[j,i], color='white')
        else:
            G.add_edge(" " + ug2[0][j], ug1[0][i], weight=weight_scale*Wug[j,i], color='blue')




edges = G.edges()
weight= [G[u][v]['weight'] for u,v in edges]
color= [G[u][v]['color'] for u,v in edges]
#pos = nx.bipartite_layout(G, g1[0])
pos = {}
for i in range(K):
    pos[ug1[0][i]] = [-1, 1 - i*0.1]
    pos[" " + ug2[0][i]] = [1, 1 - i*0.1]


stmap = dict(zip(np.hstack([ug1[0][:K],  [" "+ug2[0][i] for i in range(K)]]),  
                 100*np.log(np.round(np.hstack([ug1[1][:K], ug2[1][:K]]).astype(float)*1e5) )   ))
stmap[ug1[0][0]] /= 3
stmap[" " + ug2[0][0]] /= 3

node_size = []
node_color= []
for node in G.nodes:
    node_size.append(stmap[node])
    node_color.append(stmap[node])


fig = plt.figure()
nx.draw(G, pos = pos,node_size=2500 , node_color=node_color, edge_color=color, with_labels=True, alpha=0.5, width=weight, cmap=plt.cm.bwr)
fig.show()




