q = [144,25,33,435,6,76,3,2,354,32]

q2 = list(enumerate(q))
print q2

ss = sorted(q2, key=lambda (a,b):b)

print ss