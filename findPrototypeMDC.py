import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def load_images(directory):
    one_dim = []
    class_names = []
    labels = []
    for filename in os.listdir(directory):
        path = directory + "/" + filename
        t = filename.split('_')
        t = int(t[0])
        if t not in class_names:
            class_names.append(t)
        labels.append(t)
        im = mpimg.imread(path)
        temp = []
        for row in im:
            for w in row:
                temp.append(int(w))
        one_dim.append(temp)

    return one_dim, labels, class_names


def load_all(directory):
    train_data, label_train, class_name = load_images(directory + '/Train')
    test_data, label_test, class_name = load_images(directory + '/Test')
    return train_data, test_data, label_train, label_test, class_name


def show(images, rows, columns):
    counter = 1
    fig = plt.figure(figsize=(60, 60))
    for img in images:
        two_d = []
        for i in range(59):
            two_d.append(img[i * 60:(i + 1) * 60])
        fig.add_subplot(rows, columns, counter)
        counter += 1
        plt.imshow(two_d, cmap=plt.get_cmap('gray'))
    plt.show()



def findPrototypeData(train, label, nClasses, nFeatures):
    """ Calculate some sums of all averages for our centroids, we'll remove the test case one later """
    # create our centroidData list (format: [classID][Feat/Samples][if Feat: feat#])
    centroidData = [[0 for i in range(0, nFeatures)] for i in range(0, nClasses)]
    numberClass = [0 for i in range(0, nClasses)]
    for i in range(0, len(train)):
        for j in range(len(centroidData[0])):
            centroidData[label[i]][j] = centroidData[label[i]][j] + train[i][j]
        numberClass[label[i]] += 1
    for i in range(0, nClasses):
        centroidData[i] = np.divide(np.array(centroidData[i]), np.array(numberClass[i]))
    return centroidData

class MDCClassifier:
    """
		The PatternClassifier class holds all information and functions relevant
        to finding a sample's predicted class based on test data given. The class
        stores the test data, the number of classes to compare to, the number of
        features of each sample and the already predicted test cases
	"""
    title = ""  # the title of our database
    nClasses = 0  # the number of class types
    cNames = []  # the names of the class types (used for output)
    nFeatures = 0  # the number of features
    trainData = []  # a list of all the training data
    testData = []  # a list of all test data
    labelTrain = []  # a list of all label for train set
    labelTest = []  # a list of all label for test set
    prototypeData = []  # a list of class information used to calculate the prototype
    correct = []  # a list of sums of correct predictions (format: [[correct, attempted]...])
    predicted = []  # the compiled list of predictions (format: [sampleID, actualClass, predictedClass])

    def __init__(self, title="", nClasses=0, cNames=[], nFeatures=0, testData=[], trainData=[], labelTrain=[],
                 labelTest=[]):
        """ Initialize variables of the class to defaults/set values """
        self.title = title
        self.nClasses = nClasses
        self.cNames = cNames
        self.nFeatures = nFeatures
        self.trainData = trainData
        self.testData = testData
        self.labelTest = labelTest
        self.labelTrain = labelTrain
        for i in range(0, self.nClasses):
            self.correct.append([0, 0])
        self.correct.append([0, 0])
        self.prototypeData = findPrototypeData(self.trainData, self.labelTrain, self.nClasses, self.nFeatures)



def a():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/a")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    show(pc.prototypeData, 1, 2)

def c():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/b")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    show(pc.prototypeData, 2, 2)

def d():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/c")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    show(pc.prototypeData, 2, 5)
if __name__ == "__main__":
    a()
    # c()
    # d()
