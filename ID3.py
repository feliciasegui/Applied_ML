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

        # added attributes
        self.attributes = 0
        self.data = 0
        self.target = 0
        self.classes = 0
        self.mostc = 0


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):

        # Find best split = root node

        # value:
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

    def get_entropy(self, count_tar, classes):
        totentr = 0
        for c in classes:
            count_tar[c] +=0
            if int(count_tar[c]) != 0:
               totentr += -int(count_tar[c]) / sum(list(count_tar.values())) * math.log(int(count_tar[c]) / sum(list((count_tar.values()))), 2)
        return totentr

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self):
        # Declaring variables
        targets = self.target
        classes = self.classes
        attributes = self.attributes
        data = self.data

        count_tar = {}
        attr_entr = {}

        # Setting start values
        for c in classes:
            count_tar[c] = 0
        for attr in attributes:
            attr_entr[attr] = 0
        count_tar = dict(Counter(targets))

        # Calculating entropy
        totentr = self.get_entropy(count_tar, classes)
        print('Totala entropin: ', totentr)


        for i, attr in enumerate(attributes):

            for sample in attributes[attr]:
                count_samp = {}
                for c in classes:
                    count_samp[c] = 0

                entr_samp = 0 # entropy for each sample
                for j in range(len(data)):
                    if data[j][i] == sample:
                        count_samp[targets[j]] +=1 # counting + and - for this sample

            #Calculating entropy for this sample
                if sum(list(count_samp.values())) != 0: # if the sample actually exists we proceed with the calc

                    for c in count_samp:
                        if count_samp[c] !=0:
                            entr_samp += -count_samp[c] / sum(list(count_samp.values())) * math.log(count_samp[c] / sum(list(count_samp.values())), 2)

                    entr_samp *= sum(list(count_samp.values())) / len(data)
                    attr_entr[attr] += entr_samp

            attr_entr[attr] = totentr-attr_entr[attr]

        print('Attribut med deras information gain: ',attr_entr)
        best_attr = sorted(attr_entr, key = attr_entr.get, reverse = True)[0]

        # Change this to make some more sense
        return best_attr



    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        # Self
        self.attributes = attributes
        self.data = data
        self.target = target
        self.classes = classes

        # Root
        root = self.new_ID3_node()
        root.update({'label': None, 'attribute': None, 'entropy': self.get_entropy(dict(Counter(target)), classes), 'samples': len(target),
                         'classCounts': Counter(target), 'nodes': None })# ändrat från Counter(target).most_common()



        # If all data items belong to the same class
        if len(np.unique(target)) == 1:
            print('Vi går in i ett löv eftersom vi bara har en klass kvar')
            root.update({'label': target[0], 'classCounts': Counter(target)}) # har inte lika som sambergs här,ändrat från Counter(target).most_common()
            self.add_node_to_graph(root) # osäker på om denna ska va hör
            return root

        if len(target) == 0:
            print('Vi går in i ett löv eftersom vi inte har några targets kvar')
            root.update({'label': self.mostc})
            self.add_node_to_graph(root)
            return root

        #If no attributes are left
        if len(attributes) == 0:
            print('Vi går in i ett löv eftersom vi inte har några attributes kvar')
            mostcommon = Counter(target).most_common()
            root.update({'label' : mostcommon[0][0], 'classCounts:': Counter(target)}) # ändrat från Counter(target).most_common()
            self.add_node_to_graph(root) # osäker på om den ska va där
            return root

        # Else we do the recursion
        #else: # sambergs har inte else
            # Attribute containing max Info Gain
        target_attribute = self.find_split_attr()
        root.update({'attribute': target_attribute})

            # The attribute's samples
        sample_data_sets = attributes[target_attribute] # a tuple with the names of the attribute's samples

            # Creating new lists of data, target and classes
        new_data = [] # a list of lists, in which every element corresponds to a sample
        new_targets = []
        idx = list(attributes.keys()).index(target_attribute) # index of attribute


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
            print(new_data_samp)
            new_targets.append(new_target_samp)
            new_data.append(new_data_samp)



        # Delete used attribute
        new_attributes = attributes.copy()
        new_attributes.pop(target_attribute)
        print('Nya attributen: ', new_attributes, '\nGamla attributen: ', attributes)

         # Most common class label in new_targets,

        count_tar = dict(zip(classes, len(classes)*[0]))
        for lis in new_targets:
            for clas in lis:
                count_tar[clas] +=1
        most_comm_tar = sorted(count_tar, key = count_tar.get, reverse = True)[0]
        self.mostc = most_comm_tar
        self.add_node_to_graph(root)


        # Recursion with all the new datasets
        for i in range(len(new_attributes)):
            child = self.fit(new_data[i], new_targets[i], new_attributes, classes)
            val = attributes[target_attribute][i]
            root.update({'Nodes': child['id']})
            child.update({'value': val})
            self.add_node_to_graph(child, root['id'])



        return root



    def predict(self, data, tree) :
        predicted = list()
        #for item in data:
            # Här ska man gå vägen längs noderna som passar samples för varje item
           # for sample in item:
                # if motsvarande noden är ett löv: predicted.append.label
                # else: kör predicted med ny data och nytt träd


        return predicted