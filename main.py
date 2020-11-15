import ToyData as td
import ID3
from collections import OrderedDict
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix




def main():
    # Choose data
    #attributes, classes, data, target, data2, target2 = td.ToyData().get_data() # Toy data set
    #attributes, classes, data, target, data2, target2 = return_digits() # Digits with pixel valus 0-16
    attributes, classes, data, target, data2, target2 = return_digits_scale() # Digits with light - grey -dark scale



    id3 = ID3.ID3DecisionTreeClassifier()

    # Fit and plot tree
    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")

    # Predict
    print('Predicting...')
    predicted = id3.predict(data2, attributes)
    print(predicted)

    # Classification report and confusion matrix
    print('Classification report for the ID3 classifier: ')
    print(classification_report(list(target2), list(predicted)))
    print(confusion_matrix(list(target2), list(predicted)))




def return_digits():
    # Load and split digits data set
    digits = datasets.load_digits()

    num_split = int(0.7 * len(digits.data))
    X_train = digits.data[:num_split]  # train data (features)
    X_test = digits.data[num_split:]  # test data (features)
    y_train = digits.target[:num_split]  # train labels (y-values)
    y_test = digits.target[num_split:]  # test labels(y-values)

    # Creating tuples from label vectors y
    new_y_train = tuple(y_train)
    new_y_test = tuple(y_test)

    # Creating classes
    classes = tuple(np.unique(y_train))

    # Creating attributes
    attr_dig = []
    for i in range(len(X_train[0])):
        attr_dig.append(('pixel' + str(i), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    attr_dig = OrderedDict(attr_dig)

    return attr_dig, classes, X_train, new_y_train, X_test, new_y_test


def return_digits_scale():
    # Load and split digits data set
    digits = datasets.load_digits()

    num_split = int(0.7 * len(digits.data))
    X_train = digits.data[:num_split]  # train data (features)
    X_test = digits.data[num_split:]  # test data (features)
    y_train = digits.target[:num_split]  # train labels (y-values)
    y_test = digits.target[num_split:]  # test labels(y-values)

    # Changing values from pixel strength to strings in X-matrices
    new_X_train = []
    for pic in X_train:
        pic_list = []
        for pixel in pic:
            if pixel < 5.0:
                pic_list.append('dark')
            elif pixel > 10.0:
                pic_list.append('light')
            else:
                pic_list.append('grey')
        new_X_train.append((pic_list))

    new_X_test = []
    for pic in X_test:
        pic_list = []
        for pixel in pic:
            if pixel < 5.0:
                pic_list.append('dark')
            elif pixel > 10.0:
                pic_list.append('light')
            else:
                pic_list.append('grey')
        new_X_test.append(tuple(pic_list))

    # Creating tuples from label vectors y
    new_y_train = tuple(y_train)
    new_y_test = tuple(y_test)

    # Creating classes
    classes = tuple(np.unique(y_train))

    # Creating attributes
    attr_dig = []
    for i in range(len(X_train[0])):
        attr_dig.append(('pixel' + str(i), ['dark', 'light', 'grey']))
    attr_dig = OrderedDict(attr_dig)

    return attr_dig, classes, new_X_train, new_y_train, new_X_test, new_y_test

if __name__ == "__main__": main()