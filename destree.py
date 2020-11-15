import matplotlib.pyplot as plt
from sklearn import datasets, metrics, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import graphviz


# Load and split digits data set
digits = datasets.load_digits()

num_split = int(0.7*len(digits.data))
X_train = digits.data[:num_split] # train data (features)
y_train =  digits.target[:num_split] # train labels (y-values)
X_test = digits.data[num_split:] # test data (features)
y_test= digits.target[num_split:] # test labels(y-values)

# Fit classifier
dtree = DecisionTreeClassifier()
# min_samples_leaf sets the lowest amount of samples per leaf - a high number will result in bad prediction
# min_samples_split sets the lowest amount of samples per leaf that will split the node
dtree.fit(X_train, y_train)

# Visualize decision tree with graphviz
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render', view=True)

# With matplotlib
plt.figure()
tree.plot_tree(dtree)
plt.show()

plt.figure(figsize=(25,5))
tree.plot_tree(dtree, max_depth=2)
plt.show()

# Predict
predictions = dtree.predict(X_test)

# Evaluate
print('Classification report for the decision tree classifier')
print(classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
