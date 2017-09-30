#!/usr/bin/python

from sklearn import tree

# Smooth = 1
# Bumpy = 0
# [weight (g), Bumpy/Smooth]
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Apple = 0
# Orange = 1
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()

clf.fit(features, labels)

print clf.predict([[160, 0]])
