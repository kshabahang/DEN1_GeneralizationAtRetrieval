import spacy
import pickle
import numpy as np


root_mem = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc/FRENCH/"

#TODO use restricted vocab after pruning


nlp = spacy.load("fr_core_news_lg")


f = open(root_mem + "bgs_by_tag.pkl", "rb")
bgsByTag = pickle.load(f)
f.close()

K = 1000 #top bgs to examine


###first we sample adjective-noun bgs 
frq, bg = zip(*bgsByTag[('ADJ', 'NOM')])
adj, nom = zip(*bg)

NOMS_adj = []
ADJ_adj = []
for i in range(K):
    w = nlp(nom[i])[0]
    tag = w.tag_
    if '=' in tag:
        tag = tag.split('|')

        tag = {tag[j].split('=')[0]:tag[j].split('=')[1] for j in range(len(tag))}
        if 'NOUN__Gender' in tag and 'Number' in tag:
            if tag['NOUN__Gender'] == 'Masc' and tag['Number'] == 'Sing':
                #print(tag, det[i], w)
                ADJ_adj.append(adj[i])
                NOMS_adj.append(w.text)

###then we sample determiner-noun bgs
frq, bg = zip(*bgsByTag[('DET:ART', 'NOM')])
det, nom = zip(*bg)

NOMS_det = []
DET_det = []
for i in range(K):
    w = nlp(nom[i])[0]
    tag = w.tag_
    if '=' in tag:
        tag = tag.split('|')

        tag = {tag[j].split('=')[0]:tag[j].split('=')[1] for j in range(len(tag))}
        if 'NOUN__Gender' in tag and 'Number' in tag:
            if tag['NOUN__Gender'] == 'Masc' and tag['Number'] == 'Sing' and w.text in NOMS_adj:
                #print(tag, det[i], w)
                DET_det.append(det[i])
                NOMS_det.append(w.text)

####we also sample possesive-noun bgs
frq, bg = zip(*bgsByTag[('DET:POS', 'NOM')])
pos, nom = zip(*bg)

NOMS_pos = []
POS_pos = []
for i in range(K):
    w = nlp(nom[i])[0]
    tag = w.tag_
    if '=' in tag:
        tag = tag.split('|')

        tag = {tag[j].split('=')[0]:tag[j].split('=')[1] for j in range(len(tag))}
        if 'NOUN__Gender' in tag and 'Number' in tag:
            if tag['NOUN__Gender'] == 'Masc' and tag['Number'] == 'Sing' and w.text in NOMS_adj:
                #print(tag, det[i], w)
                POS_pos.append(pos[i])
                NOMS_pos.append(w.text)



####we also sample possesive-noun bgs
frq, bg = zip(*bgsByTag[('PRO:DEM', 'NOM')])
dem, nom = zip(*bg)

NOMS_dem = []
DEM_dem = []
for i in range(K):
    w = nlp(nom[i])[0]
    tag = w.tag_
    if '=' in tag:
        tag = tag.split('|')

        tag = {tag[j].split('=')[0]:tag[j].split('=')[1] for j in range(len(tag))}
        if 'NOUN__Gender' in tag and 'Number' in tag:
            if tag['NOUN__Gender'] == 'Masc' and tag['Number'] == 'Sing' and w.text in NOMS_adj:
                #print(tag, det[i], w)
                DEM_dem.append(dem[i])
                NOMS_dem.append(w.text)


###now we need to match determiner/posessive/demonstratives with adjectives, based on noun
choice = np.zeros(3)
for i in range(len(NOMS_adj)):
    choice[0] = int( NOMS_adj[i] in NOMS_pos )
    choice[1] = int( NOMS_adj[i] in NOMS_dem )
    choice[2] = int( NOMS_adj[i] in NOMS_det )
    if sum(choice) == 1:
        idx = np.argmax(choice)
    elif sum(choice) == 2:
        idx_neg = np.argmin(choice)

    print(inPOS, inDEM, inDET)



