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


def normalize(data):
    """ Create a list of normalized data based off of min/max values and a dataset"""
    # lets make sure everything in the array is a float..
    for i in range(0, len(data)):
        data[i] = list(map(float, data[i]))
    # rotate the array for easy access to min/max
    rotatedArray = list(zip(*data[::-1]))
    # loop through each feature, find the min/max of that feature line, normalize the data on that line, repeat
    for i in range(2, len(rotatedArray)):
        minVal = min(rotatedArray[i])
        maxVal = max(rotatedArray[i])
        # traverse through each sample in list, i is the feature, j is the sample
        for j in range(0, len(rotatedArray[i])):
            # be careful if max or min equal each other, then we are dividing by 0
            if (maxVal - minVal != 0):
                data[j][i] = (data[j][i] - minVal) / (maxVal - minVal)
    return data


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


def euclidean_distance(sample1, sample2):
    temp = 0
    for i in range (len(sample1)):
        temp+=  (sample1[i] - sample2[i]) ** 2
    return temp


def findPrediction(sample, protypeData,cName):
    dist = []
    for i in range(len(protypeData)):
        dist.append([cName[i],euclidean_distance(sample,protypeData[i])])
    dist.sort(key=lambda tup: tup[1])
    return dist[0][0]

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
        self.trainData = normalize(trainData)
        self.testData = normalize(testData)
        self.labelTest = labelTest
        self.labelTrain = labelTrain
        for i in range(0, self.nClasses):
            self.correct.append([0, 0])
        self.correct.append([0, 0])
        self.prototypeData = findPrototypeData(self.trainData, self.labelTrain, self.nClasses, self.nFeatures)


    def evaluate(self):
        true_answer =0
        print("start evaluation MDC Classifier with Test data")
        for i in range(len(self.testData)):
            prediction = int(findPrediction(self.testData[i], self.prototypeData,self.cNames))
            if self.labelTest[i] == prediction:
                true_answer += 1
            print("class : "+str(self.labelTest[i])+ " predicted : "+ str(prediction))
        print("accuracy : "+ str(true_answer/len(self.labelTest)*100) + "%")
        return true_answer/len(self.labelTest)

def b():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/a")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    pc.evaluate()
def c():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/b")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    pc.evaluate()
def d():
    train_data, test_data, label_train, labels_test, class_names = load_all(
        "D:/univesity/foqelisans/pattern_recognition/SPR_HW2/code/Q4/Dataset/c")
    pc = MDCClassifier(title="01 train set", nClasses=len(class_names), cNames=class_names,
                           nFeatures=len(train_data[0]), labelTrain=label_train,
                           labelTest=labels_test, trainData=train_data, testData=test_data)
    pc.evaluate()




if __name__ == "__main__":
    b()
    # c()
    # d()