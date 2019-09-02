import csv

preds = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[]
}

with open('pred_val_multinomial.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter = ' ')
    for row in reader:
        pred = int(row[-1])
        preds[pred].append(float(row[pred + 1]))

upper = {}
for c in preds:
    l = sorted(preds[c])
    print 'class %d: [%.3f, %.3f]' % (c, l[0], l[-1])
    upper[c] = l[-1]

print upper

binClasses = ['hornpipe', 'jig', 'other44', 'polka',
              'reel', 'slide', 'slipjig', 'waltz']
refClasses = dict.fromkeys(range(8), 0)
with open('ref_multinomial.csv', 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        refClasses[ binClasses.index(row[1]) ] += 1
print refClasses

preds = {
    0:[],
    1:[]
}

with open('pred_val_binomial.csv', 'r') as infile:
    reader = csv.reader(infile, delimiter = ' ')
    for row in reader:
        pred = int(row[-1])
        preds[pred].append(float(row[pred + 1]))

for c in preds:
    l = sorted(preds[c])
    print 'class %d: [%.5f, %.5f]' % (c, l[0], l[-1])
