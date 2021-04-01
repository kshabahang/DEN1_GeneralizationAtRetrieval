from progressbar import ProgressBar
import pickle



root_mem = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc/FRENCH/"

f = open(root_mem + "wikipediaFR-2008-06-18.tag", "r",  encoding='latin-1')
raw = f.readlines()[:34779381]
f.close()


text = []
tags = []

for i in range(len(raw)):
    if '\t' in raw[i]:
        [word, tag, junk] = raw[i].strip('\n').split('\t')
        text.append(word)
        tags.append(tag)

bgs_by_tag = {}
check = 0
pbar = ProgressBar(maxval=len(text)).start()
for i in range(len(text)-1):
    [w1,w2] = text[i:i+2]
    [t1,t2] = tags[i:i+2]
    w1 = w1.lower()
    w2 = w2.lower()
    condMet = False
    if (t1,t2) in bgs_by_tag:
        if (w1, w2) in bgs_by_tag[(t1,t2)]:
            bgs_by_tag[(t1,t2)][(w1,w2)] += 1
        else:
            bgs_by_tag[(t1,t2)][(w1,w2)] = 1
            condMet = True
    else:
        bgs_by_tag[(t1,t2)] = {(w1,w2):1}
        condMet = True
    check += int(condMet) 
    pbar.update(i+1)


for tag in bgs_by_tag.keys():
    bg, frq = zip(*bgs_by_tag[tag].items())
    bgs_by_tag[tag] = sorted(zip(frq, bg))[::-1]

f = open(root_mem + "bgs_by_tag.pkl", "wb")
pickle.dump(bgs_by_tag, f)
f.close()

f = open(root_mem + "FRENCH.txt", "w")
for i in range(len(text)):
    if text[i] == '.':
        f.write(text[i] + '\n')
    else:
        f.write(text[i] + ' ')
f.close()
