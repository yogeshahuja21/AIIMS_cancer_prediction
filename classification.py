from typing import Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
pat: Union[Union[TextFileReader, DataFrame], Any] = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/PycharmProjects/ml_proj_healthcare/cancer1.csv")
print(pat.head())
print(pat.shape)
print(pat['diagnosis'].unique())
print(pat.groupby('diagnosis').size())
sns.countplot(pat['diagnosis'],label="Count")
plt.show()
pat.drop('diagnosis', axis=1).boxplot()
plt.savefig('cancer Classification')
plt.show()
feature_names = ['texture_mean', 'area_mean', 'compactness_mean', 'concave points_mean']
X = pat[feature_names]
y = pat['diagnosis']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('diagnosis_scatter_matrix')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

print("#########################################")
print("    DECISION TREE       ")
def importdata():
    pat = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/PycharmProjects/ml_proj_healthcare/cancer1.csv")

    return pat


def splitdataset(pat):
    X = pat.values[:, 2:24]
    Y = pat.values[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_using_entropy(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values are:")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print("Report : ", classification_report(y_test, y_pred))


def main():
    data=importdata()
    X, Y, X_train, X_test, y_train,y_test=splitdataset(data)


    clf_gini=train_using_gini(X_train, X_test, y_train)

    clf_entropy=train_using_entropy(X_train, X_test, y_train)
    y_pred_gini=prediction(X_test,clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    y_pred_entropy=prediction(X_test,clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__== "__main__":
    main()

print("#########################################")
print("    LOGISTIC REGRESSION       ")

def init():
    df=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/PycharmProjects/ml_proj_healthcare/cancer1.csv")
    df = df.drop("id", 1)

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    train, test = train_test_split(df, test_size=0.3, random_state=1)
    # print("Train data hai ye")
    # print(train)
    # print("Test data hai ye")
    # print(test)

    #yhan pe train x aur train y banaya hai

    train_x = train.loc[:, 'radius_mean': 'fractal_dimension_worst']
    train_y = train.loc[:, ['diagnosis']]

    #yhan pe test x aur test y banaya hai
    test_x = test.loc[:, 'radius_mean': 'fractal_dimension_worst']
    test_y = test.loc[:, ['diagnosis']]

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)


    # print("Train x data hai ye")
    # print(train_x)
    # print("Train y data hai ye")
    # print(train_y)
    # print("Test x data hai ye")
    # print(test_x)
    # print("Test y data hai ye")
    # print(test_y)

    d = model(train_x.T, train_y.T, num_of_iterations=10000, alpha=0.000001)

    costs = d["costs"]
    w = d["w"]
    b = d["b"]

    #  plot between cost and number of iterations banaya hai
    plt.plot(costs)
    plt.title("Cost vs #Iterations")
    plt.xlabel("Number of Iterations ( * 10)")
    plt.ylabel("Cost")
    plt.show()
    plt.savefig("Cost vs iterations")

    # calculating the accuracy on Training and Test Data
    Y_prediction_train = predict(train_x.T, w, b)
    Y_prediction_test = predict(test_x.T, w, b)
    print("this is predicted output")
    print(Y_prediction_test)

    print("\nTrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y.T)) * 100))

    print("\nTest accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y.T)) * 100))


#ye weights aur bias initialize krne ke liye
def initialize(m):
    w = np.zeros((m, 1))
    b = 0

    return w, b
#function for calculating sigmoid of x
def sigmoid(X):
    return 1/(1+ np.exp(- X))

#forward aur backward propagation ke liye function hai
def propogate(X, Y, w, b):
    m = X.shape[1] #number of training examples

    #calculating the cost in forward propagation
    Z = np.dot(w.T,X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y* np.log(A) + (1-Y) * np.log(1 - A))


    #calculating the gradients in backward propagation
    dw = (1/m)* np.dot(X, (A - Y).T)
    db = (1/m)* np.sum(A - Y)

    grads = {"dw":dw, "db":db}

    return grads, cost



#gradient descent krne ke liye function hai
def optimize(X, Y, w, b, num_of_iterations, alpha):
    costs = []
    for i in range(num_of_iterations):
        grads, cost = propogate(X, Y, w, b)

        dw = grads["dw"]
        db = grads["db"]

        w = w- alpha * dw
        b = b- alpha * db

        # Storing the cost at interval of every 10 iterations
        if i%10 == 0:
            costs.append(cost)
            # print("cost after %i iteration : %f" % (i, cost))

    parameters = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return parameters, grads, costs


#predictions on the data set (mapping probabilities to 0 or 1) karne ke liye
def predict(X, w, b):
    m = X.shape[1]  #no.of training examples

    y_prediction = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if (A[0, i] < 0.5):
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1

    return y_prediction


def model(Xtrain, Ytrain, num_of_iterations, alpha):
    # print("ye model ka xtrain hai")
    # print(Xtrain)
    # print("ye model ka ytrain hai")
    # print(Ytrain)
    dim = Xtrain.shape[0]
    # print(dim)#ye no of features in x train hai

    w, b = initialize(dim)
    # print("ye model ka w hai")
    # print(w)
    # print("ye model ka b hai")
    # print(b)

    parameters, grads, costs = optimize(Xtrain, Ytrain, w, b, num_of_iterations, alpha)
    w = parameters["w"]
    b = parameters["b"]

    d = {"w": w, "b": b, "costs": costs}

    return d
init()
print("Random Forest Algorithm on Healthcare Dataset")
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd

# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	# dataset=dataset.pop(0)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'C:/Users/Yogesh Kumar Ahuja/PycharmProjects/ml_proj_healthcare/cancer.csv'
dataset = load_csv(filename)
# convert string attributes to float
for i in range(0, len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
	scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


print("######## K MEANS ALGORITHM ###########")

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
data=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/PycharmProjects/ml_proj_healthcare/cancer1.csv")
x = data.values[:, 2:24]

y = data.values[:, 1]
for i in range(1,10):
    km=KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=5,random_state=0)
    yk=km.fit(x)
    print(yk)
    y=list(km.fit_predict(x))
    print(type(y))
    # kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # y_kmeans = kmeans.fit_predict(x)
    # plt.scatter()
    # x.append(km.inertia_)
    print(x)
    print(km.n_clusters)
    print(km.cluster_centers_)
# fig=plt.figure(figsize=(5,5))
# fig.add_subplots(1,1,1,facecolor="1.0")
# colmap={1:'r',2:'g',3:'b'}
# y1=[1]
# y2=[2]
c_old=np.zeros(x.shape)
# y1=[i for i in y if(i==0)]
# print(y1)
# y2=[i for i in y if(i==1)]
# print(y2)

# x=zip(y1,y2)
# print(list(x))
#
# c=zip(y1,y2)
points=np.array(x[m] for m in range(len(x)))
clusters =np.zeros(len(x))

fig,ax=plt.subplots()
# for i in range(2):
#     ax.scatter(points[:,0],points[:,1])
#
#
# ax.scatter(y1,y2)
# plt.show()