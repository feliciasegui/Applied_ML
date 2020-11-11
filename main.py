import ToyData as td
import ID3
import ID3copy

import numpy as np
from sklearn import tree, metrics, datasets


def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()


    id3 = ID3.ID3DecisionTreeClassifier()
    #id3 = ID3copy.ID3DecisionTreeClassifier()

    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, myTree)
    print(predicted)

    best_attr = id3.find_split_attr()
    print(best_attr)

if __name__ == "__main__": main()