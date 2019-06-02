import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def init():
    df=pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/cancer.csv")
    df = df.drop("id", 1)
    # df = df.drop("Unnamed: 32", 1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    train, test = train_test_split(df, test_size=0.3, random_state=1)
    print("Train data hai ye")
    print(train)
    print("Test data hai ye")
    print(test)

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


    print("Train x data hai ye")
    print(train_x)
    print("Train y data hai ye")
    print(train_y)
    print("Test x data hai ye")
    print(test_x)
    print("Test y data hai ye")
    print(test_y)

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

    # calculating the accuracy on Training and Test Data
    Y_prediction_train = predict(train_x.T, w, b)
    Y_prediction_test = predict(test_x.T, w, b)

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
    Z = np.dot(w.T,X) + b;
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
            print("cost after %i iteration : %f" % (i, cost))

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
    print("ye model ka xtrain hai")
    print(Xtrain)
    print("ye model ka ytrain hai")
    print(Ytrain)
    dim = Xtrain.shape[0]
    print(dim)#ye no of features in x train hai

    w, b = initialize(dim)
    print("ye model ka w hai")
    print(w)
    print("ye model ka b hai")
    print(b)

    parameters, grads, costs = optimize(Xtrain, Ytrain, w, b, num_of_iterations, alpha)
    w = parameters["w"]
    b = parameters["b"]

    d = {"w": w, "b": b, "costs": costs}

    return d
init()

