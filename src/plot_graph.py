from matplotlib import pyplot as plt
import networkx as nx
import pickle


f = open("top_weights.pkl", "rb")
top_weights = pickle.load(f)
f.close()


(ws1, ws2, sts1, sts2, weights) = top_weights[-1]

G = nx.Graph()
for i in range(100):
    for j in range(100):
        if weights[i,j] != 0:
            G.add_edge( ws1[i], " "+ws2[j])

pos = nx.bipartite_layout(G, ws1) 
nx.draw(G, pos = pos, node_color="white", with_labels=True, alpha=0.5)
plt.show()
