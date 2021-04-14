from matplotlib import pyplot as plt
import networkx as nx
import pickle
import sys


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

G = nx.Graph()

g1 = list(g1[0][:20])
g2 = list(g2[0][:20])
