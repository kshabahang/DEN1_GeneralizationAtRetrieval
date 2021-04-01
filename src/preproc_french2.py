root_mem = "/home/ubuntu/LTM/DEN1_GeneralizationAtRetrieval/rsc/FRENCH/"

f = open(root_mem + "FRENCH.txt", "r")
raw = f.readlines()
f.close()


f = open(root_mem + "FRENCH_clean.txt", "w")

for i in range(len(raw)):
    if '|' not in raw[i] and '( Comment ? )' not in raw[i]:
        f.write(raw[i].lower())
f.close()
