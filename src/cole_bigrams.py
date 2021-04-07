import spacy
import pickle
import numpy as np
from progressbar import ProgressBar


root_mem = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc/FRENCH/"

#TODO use restricted vocab after pruning


nlp = spacy.load("fr_core_news_lg")


f = open(root_mem + "bgs_by_tag.pkl", "rb")
bgsByTag = pickle.load(f)
f.close()


f = open("french_nouns.txt", "r")
nouns=  f.readlines()
f.close()
nouns = list(filter(lambda s : 'mas' in s, nouns)) #only masculine
#noun_sing, noun_plur = zip(*[(nouns[i], nouns[i+1]) for i in range(0, int(len(nouns)/2), 2)])
#noun_sing = [noun_sing[i].strip().lower().split(';')[0] for i in range(len(noun_sing))]
#noun_plur = [noun_plur[i].strip().lower().split(';')[0] for i in range(len(noun_plur))]
#noun_sing2plur = dict(zip(noun_sing, noun_plur))

f = open("french_adjectives.txt", "r")
adjectives=  f.readlines()
f.close()
adjectives = list(filter(lambda s : ('sg' in s or 'pl' in s) and ('epi' not in s), adjectives))
#adj_sing, adj_plur = zip(*[tuple(adjectives[i:i+2]) for i in range(0, int(len(adjectives)/2), 2)])
#adj_sing = [adj_sing[i].strip().lower().split(';')[0] for i in range(len(adj_sing))]
#adj_plur = [adj_plur[i].strip().lower().split(';')[0] for i in range(len(adj_plur))]
#adj_sing2plur = dict(zip(adj_sing, adj_plur))

adj_mas2fem = {}
adj_sing2plur = {}
print("Mapping adjectives to plural and feminine forms")
pbar= ProgressBar(maxval=len(adjectives)).start()
for i in range(len(adjectives)):
    if 'fem' in adjectives[i] and 'sg' in adjectives[i]:
        adj = adjectives[i].split(';')[0]
        lemma = nlp(adj)[0].lemma_
        tag = nlp(lemma)[0].tag_
        if "Mas" in tag:
            adj_mas2fem[lemma] = adj
    elif ';pl' in adjectives[i]:
        adj = adjectives[i].split(';')[0]
        lemma = nlp(adj)[0].lemma_
        tag = nlp(lemma)[0].tag_
        if "Mas" in tag:
            adj_sing2plur[lemma] = adj
    pbar.update(i+1)



f = open("french_pronouns.txt", "r")
pronouns=  f.readlines()
f.close()

f = open("french_determiners.txt", "r")
determiners =  f.readlines()
f.close()

determiners = pronouns + determiners

print("Mapping determiners to plural and feminine forms")
pbar = ProgressBar(maxval=len(determiners)).start()
det_mas2fem = {}
det_sing2plu= {}
for i in range(len(determiners)):
    if ';pl' in determiners[i]:
        pro = determiners[i].split(';')[0]
        lemma = nlp(pro)[0].lemma_
        if lemma != pro:
            det_sing2plu[lemma] = pro
    if ';fem;' in determiners[i]:
        pro =  determiners[i].split(';')[0]
        lemma = nlp(pro)[0].lemma_
        if lemma != pro:
            det_mas2fem[lemma] = pro
    pbar.update(i+1)



NOM_adj = []
ADJ_adj = []
ADJ_adj_plur = []
ADJ_adj_fem  = []
print("Collecting adjective-noun pairs")
K = 5000
pbar = ProgressBar(maxval=K).start()
for i in range(K):
    (fr, (adj, nom)) = bgsByTag[('ADJ', 'NOM')][i]
    if nom[0] not in 'auouwyq':
        w_nom = nlp(nom)[0]
        toSkip = False
        if 'Masc' in w_nom.tag_:
            if 'Sing' in w_nom.tag_:
                adj_txt = adj
                if adj_txt not in adj_sing2plur:
                    toSkip = True
                else:
                    adj_txt_plur = adj_sing2plur[adj_txt]
                nom_txt = nom
            else:
                #singularize
                adj_txt_plur = adj
                adj_txt = nlp(adj)[0].lemma_
                nom_txt = w_nom.lemma_
            try:
                if adj_txt == adj_txt_plur:
                    toSkip = True
            except:
                toSkip = True

            if adj_txt not in adj_mas2fem:
                toSkip = True

            if not toSkip:
                NOM_adj.append(nom_txt)
                ADJ_adj.append(adj_txt)
                ADJ_adj_plur.append(adj_txt_plur)
                ADJ_adj_fem.append(adj_mas2fem[adj_txt])
            
print("Mapping nouns to their determiners etc.")
frq, bgs = zip(*bgsByTag[('DET:ART', 'NOM')])
det, nom = zip(*bgs)
nom2det = {}
for i in range(len(nom)):
    if nom[i] in nom2det:
        if det[i] not in nom2det[nom[i]]:
            nom2det[nom[i]].append(det[i])
    else:
        nom2det[nom[i]] = [det[i]]

frq, bgs = zip(*bgsByTag[('DET:POS', 'NOM')])
pos, nom = zip(*bgs)
nom2pos = {}
for i in range(len(nom)):
    if nom[i] in nom2pos:
        if pos[i] not in nom2pos[nom[i]]:
            nom2pos[nom[i]].append(pos[i])
    else:
        nom2pos[nom[i]] = [pos[i]]

frq, bgs = zip(*bgsByTag[('PRO:DEM', 'NOM')])
dem, nom = zip(*bgs)
nom2dem = {}
for i in range(len(nom)):
    if nom[i] in nom2dem:
        if dem[i] not in nom2dem[nom[i]]:
            nom2dem[nom[i]].append(dem[i])
    else:
        nom2dem[nom[i]] = [dem[i]]

print("Constructing stimulus groups")
nom2ctx = {}
pbar = ProgressBar(maxval=len(NOM_adj)).start()
for i in range(len(NOM_adj)):
    nom = NOM_adj[i]
    adj = ADJ_adj[i]
    adj_plur = ADJ_adj_plur[i]
    adj_fem = ADJ_adj_fem[i]
    dets = []
    if nom in nom2det:
        dets += nom2det[nom]
    if nom in nom2pos:
        dets += nom2pos[nom]
    if nom in nom2dem:
        dets += nom2dem[nom]
    ###we only use those for which we have the feminine and plural forms
    dets_keep = []
    for j in range(len(dets)):
        if dets[j] in det_mas2fem and dets[j] in det_sing2plu:
            dets_keep.append(dets[j])
    if len(dets_keep) > 0:
        idx = np.random.randint(len(dets_keep))
        det = dets_keep[idx]
        nom2ctx[nom] = {"open":{"control": adj, "number": adj_plur, "gender": adj_fem}, 
                        "close":{"control": det, "number":det_sing2plu[det],"gender":det_mas2fem[det]}}
    pbar.update(i+1)
        
f = open("cole_stims.pkl", "wb")
pickle.dump(nom2ctx, f)
f.close()




        


