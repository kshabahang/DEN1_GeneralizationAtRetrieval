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

K = 10000 #top bgs to examine

f = open("french_nouns.txt", "r")
nouns=  f.readlines()
f.close()
nouns = list(filter(lambda s : 'mas' in s, nouns)) #only masculine
noun_sing, noun_plur = zip(*[(nouns[i], nouns[i+1]) for i in range(0, int(len(nouns)/2), 2)])
noun_sing = [noun_sing[i].strip().lower().split(';')[0] for i in range(len(noun_sing))]
noun_plur = [noun_plur[i].strip().lower().split(';')[0] for i in range(len(noun_plur))]
noun_sing2plur = dict(zip(noun_sing, noun_plur))

f = open("french_adjectives.txt", "r")
adjectives=  f.readlines()
f.close()
adjectives = list(filter(lambda s : ('sg' in s or 'pl' in s) and ('epi' not in s), adjectives))
adj_sing, adj_plur = zip(*[tuple(adjectives[i:i+2]) for i in range(0, int(len(adjectives)/2), 2)])
adj_sing = [adj_sing[i].strip().lower().split(';')[0] for i in range(len(adj_sing))]
adj_plur = [adj_plur[i].strip().lower().split(';')[0] for i in range(len(adj_plur))]
adj_sing2plur = dict(zip(adj_sing, adj_plur))

f = open("french_pronouns.txt", "r")
pronouns=  f.readlines()
f.close()

f = open("french_determiners.txt", "r")
determiners =  f.readlines()
f.close()


NOM_adj = []
ADJ_adj = []
ADJ_adj_plur = []
for i in range(5000):
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
            if not toSkip:

                print("/".join(["le", "les"]), "/".join([adj_txt, adj_txt_plur]), nom_txt)
            

















        


