import json
with open('QA_train.json') as dataset:
    l = dataset.readline()
    jd = json.loads(l.strip())
    cc = len(jd)



ll = 0




