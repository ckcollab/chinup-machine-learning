'''
Trying to make a XOR neural net using sklearn
'''

from sklearn import tree


input = [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0]
]
target = [
    [1],
    [0],
    [1],
    [0],
]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(input, target)
print clf.predict([[0, 0]])  # outputs 0
print clf.predict([[1, 0]])  # outputs 1
