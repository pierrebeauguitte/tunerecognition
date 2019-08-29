import csv
import matplotlib.pyplot as plt

ref = {}
with open('ref_binomial.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter = ',')
    for row in reader:
        ref[int(row[0])] = 0 if row[1] == 'compound' else 1

positive = 0 # compound
# positive = 1 # simple

nCompound = 0
nSimple = 0

pred = {}
with open('pred_val_binomial.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter = ' ')
    for row in reader:
        pred[row[0]] = float(row[1 + positive])
        if ref[int(row[0][:3])] == 0:
            nCompound += 1
        else:
            nSimple += 1

print '%d compound, %d simple' % (nCompound, nSimple)
        
thresholds = map(lambda x: x/100., range(1, 100))

P = {}
R = {}
F1 = []

def f(p, r, beta = 1):
    if p == 0 and r == 0:
        return 0
    return (1 + beta**2) * (p*r) / ((beta**2)*p + r)

for t in thresholds:

    TP_FN = 0   # 0=c, 1=s
    TP = 0
    FP = 0

    for p, probs in pred.iteritems():
        trueLabel = ref[int(p[:3])]
        if trueLabel == positive:
            TP_FN += 1
        if probs > t:
            if trueLabel == positive:
                TP += 1
            else:
                FP += 1

    P[t] = TP / float(TP + FP) if (TP+FP > 0) else 0
    R[t] = TP / float(TP_FN)
    F1.append(f(P[t], R[t], beta = 1))
    
maxF = max(F1)
index = thresholds[F1.index(maxF)]

print "Max F1 score %.3f at %.3f" % (maxF, index)

plt.figure(figsize=(5, 4))
plt.plot(thresholds, F1, label="F1 score")
plt.plot(thresholds, [P[s] for s in thresholds], linewidth = 0.7, label = "precision")
plt.plot(thresholds, [R[s] for s in thresholds], linewidth = 0.7, label = "recall")
plt.legend()
plt.stem([index], [maxF])
plt.savefig("figures/maxF1.pdf", bbox_inches='tight')
