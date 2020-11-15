from collections import Counter
from graphviz import Digraph
import numpy as np
from numpy import*
from collections import Counter



class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

        # Attributes
        self.attributes = 0
        self.data = 0
        self.target = 0
        self.classes = 0
        self.mostc = 0

        # Root
        self.root = 0
        self.entropy = 0


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):

        # Find best split = root node
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    def get_entropy(self, count_beg, classes):
        totentr = 0
        count_tar = {}

        # Make sure that all classes exist in count_tar, at least with the value 0
        for c in classes:
            count_tar[c] = 0
        count_tar.update(count_beg)

        # Calculating the total entropy
        for c in classes:
            if int(count_tar[c]) != 0:
                totentr += -int(count_tar[c]) / sum(list(count_tar.values())) * math.log(
                    int(count_tar[c]) / sum(list((count_tar.values()))), 2)
        return totentr


    def find_split_attr(self):

        # Targets and data
        targets = self.target
        data = self.data

        # Classes
        classes = self.classes

        # Attributes, ordered dictionary with attribute as key and possible values in a tuple, as value
        attributes = self.attributes

        # Total entropy
        totentr = 0

        # Count occurence of each class, in targets
        count_tar = {}

        for c in classes:
            count_tar[c] = 0

        for target in targets:
            count_tar[target] += 1


        # Calculating entropy
        for c in classes:
            if int(count_tar[c]) != 0:
             totentr += -int(count_tar[c]) / sum(list(count_tar.values())) * math.log(int(count_tar[c]) / sum(list((count_tar.values()))), 2)



        # Entropy for each attribute, attribute as key and its entropy as value
        attr_entr = {}

        for attr in attributes:
            attr_entr[attr] = 0

        # Iterate through attributes(ex pixel or color, size, shape)
        for i, attr in enumerate(attributes):

            # Iterate through samples (ex pixel strength or yellow, green, large, small)
            for sample in attributes[attr]:
                count_samp = {}

                # A dictionary for counting number of each class, for a specific sample
                for c in classes:
                    count_samp[c] = 0

                entr_samp = 0 # entropy for each sample

                # Iterate through data but only looking at the element for the specific attribute
                for j in range(len(data)):
                    if data[j][i] == sample:
                        count_samp[targets[j]] +=1 # counting + and - for this sample


                #Calculating entropy for this sample
                if sum(list(count_samp.values())) != 0: # if the sample actually exists we proceed with the calc

                    #Iterating through the sample's dict of number of classes
                    for c in count_samp:
                        if count_samp[c] !=0: # If the class exists for the sample
                            entr_samp += -count_samp[c] / sum(list(count_samp.values())) * math.log(count_samp[c] / sum(list(count_samp.values())), 2)


                # Multiplying entropy with weight and adding it to the total entropy for the attribute
                entr_samp *= sum(list(count_samp.values())) / len(data) # dessa var en tab in förut
                attr_entr[attr] += entr_samp # dessa va en tab in förut
                self.entropy = attr_entr[attr]






            # Information gain for attibute
            attr_entr[attr] = totentr-attr_entr[attr]

        # Extracting the attribute with highest information gain and returning it as a string
        best_attr = sorted(attr_entr, key = attr_entr.get, reverse = True)[0]

        return best_attr



    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        # Updating self
        self.attributes = attributes
        self.data = data
        self.target = target
        self.classes = classes

        # Root
        root = self.new_ID3_node()
        root.update({'label': None, 'attribute': None, 'entropy': self.get_entropy(dict(Counter(target)), classes), 'samples': len(target),
                         'classCounts': Counter(target).most_common(), 'nodes': [] })


        #If samples is empty, add leaf node with label = most common value in samples
        if len(target) == 0:
            root.update({'label': self.mostc}) # Label is most common target
            #root.update({'entropy': 0}) # Should not be here
            self.add_node_to_graph(root) # adding node to the graph
            self.root = root # updating the root to be able to keep track
            return root

        # If all targets are same return the node (leaf)
        if target.count(target[0]) == len(target): # if the number of the most common target is the same as the length of target we only have one class
            mostcommon = Counter(target).most_common()
            root.update({'label': mostcommon[0][0]}) # the most common class
            #root.update({'entropy': 0}) # entropy should be zero as all are of the same class
            root.update({'classCounts': mostcommon}) # adding number of each class
            self.add_node_to_graph(root)
            self.root = root
            return root


        # If no attributes are left
        if len(attributes) == 0:
            mostcommon = Counter(target).most_common()
            root.update({'label':mostcommon[0][0]}) # adding most common class
            root.update({'classCounts': mostcommon}) # adding the number of each class
            self.add_node_to_graph(root) # add node to graph
            self.root = root # updating root
            return root


        # Finding the split attribute and updating the root
        splitattribute = self.find_split_attr()
        root.update({'attribute': splitattribute})

        # Updating the root's entropy as it is calculated for the splitattribute
        #root.update({'entropy': self.entropy}) # DENNA SKA BORT!!!

        # Extract new attributes
        newattributes = attributes.copy() # need to copy to keep OrderedDict
        newattributes.pop(splitattribute)

        # Length of data (number of items / images etc)
        l = len(data)

        #Number of different values in attribute (number of pixel values, colors etc)
        nbr = len(attributes[splitattribute])

        #Index of split attribute
        a_idx = list(attributes).index(splitattribute)

        #Splitted data
        new_data = []
        new_targets = []

        # Iterating through the number of samples (colors ex)
        for n in range(nbr):
            tempdata = [] # temporary list for data
            temptarget = [] # temporary list for targets

            # Iterating through targets and data
            for i in range(l):
                #If the data at index(i,attribute) == value at attribute
                if data[i][a_idx] == attributes[splitattribute][n]:
                    tlist = list(data[i]) # the whole item with its samples is copied
                    tlist.remove(data[i][a_idx]) # the sample corresponding the split attribute is removed as we do not want it in next iteration
                    datatup = tuple(tlist) # converting to tup as we want
                    tempdata.append(datatup) # adding the new tuple to the temporary data list
                    temptarget.append(target[i]) # adding the target (class) corresponding to the data-item

            # Appending the temporary data lists, each list corresponds to one of the samples corresponding to the attribute (ex if we have three colors we will have three lists in these lists)
            new_data.append(tempdata)
            new_targets.append(temptarget)


            '''         # The attribute's samples
                 sample_data_sets = attributes[target_attribute] # a tuple with the names of the attribute's samples

                 for sample in sample_data_sets:
                     new_data_samp = [] # temporary list of data that corresponds to the sample
                     new_target_samp = []
                     for i, item in enumerate(data):
                         if item[idx] == sample:

                             # Extracting and appending the item except from the value belonging to the sample
                             temp_samp = list(item)
                             temp_samp.remove(temp_samp[idx])
                             new_data_samp.append(tuple(temp_samp))

                             # Appending the target belonging to the data item
                             new_target_samp.append(target[i])

                     new_targets.append(new_target_samp)
                     new_data.append(new_data_samp)'''


            # Updating the most common class and assing to the node (I do not really understand why)
            mostcommon = Counter(target).most_common()
            self.mostc = mostcommon[0][0]
            self.add_node_to_graph(root)

        # Iterating through the number of samples = the number of possible branches, and using recursion for creating them
        for n in range(nbr):
            child = self.fit(new_data[n],new_targets[n],newattributes,classes) # the recursion part
            val = attributes[splitattribute][n] # the branch we currently are "on" , decided by the sample
            root['nodes'].append(child) # the child node is added to its mother node's list
            child.update({'value':val}) #updating the child's dictionary
            self.add_node_to_graph(child,root['id']) # adding to graph and linking the root with its child


            '''    # Recursion with all the new datasets
        for i in range(len(attributes[splitattribute])): #range(len(new_attributes)): kan ha hittat felet här
            print(len(newattributes), new_data[i], new_targets[i], newattributes, classes, attributes[splitattribute][i])
            child = self.fit(new_data[i], new_targets[i], newattributes, classes)
            val = attributes[target_attribute][i]
            root['nodes'].append(child) 
            child.update({'value': val})
            self.add_node_to_graph(child, root['id'])'''

        # updating root
        self.root = root
        return root



    def predict(self, data, attributes) :
        # List with predictions
        predicted = list()

        #Declaring root and attributes
        root = self.root
        attributes = attributes

        #Iterating through the input data
        for data in data:
            root = self.root

            # While the root has children
            while(root.get('nodes') != []):

                #Iterating through the attributes
                for i,attr in enumerate(attributes):


                    # If the root's attribute matches the attribute in the loop
                    if root.get('attribute') == attr:

                        # Extract the sample (ex pixel value, blue, green etc)
                        datavalue = data[i]

                        #Iterating through the root's children
                        for childs in root.get('nodes'):

                            # If the the child's value ("branch") is same as the datavalue we "choose" that branch
                            if childs.get('value') == datavalue:

                                # Our root is now the child
                                root = childs

            # Appending the label to the predicted list
            predicted.append(root.get('label'))
        return predicted