import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, plot_confusion_matrix, recall_score, precision_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.datasets import mnist
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score, train_test_split
import joblib
from sklearn.tree import DecisionTreeClassifier
HEIGHT = WIDTH = 28
#loading the dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#printing the shapes of the vectors
#x_train: (60000, 28, 28)
#Y_train: (60000,)
#x_test:  (10000, 28, 28)
#Y_test:  (10000,)


class Preprocessing:
    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test

    def normalization(self):
        #reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        #convert the datatype
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        #normalization into scale [0,1]
        self.x_train /= 255
        self.x_test /= 255
        return self.x_train,  self.x_test
    def use_TSNE(self):
        tsvd = TruncatedSVD(n_components=50)

        # Transform the training and test data using TSVD
        X_train_tsvd = tsvd.fit_transform(self.x_train)
        X_test_tsvd = tsvd.transform(self.x_test)

        tsne = TSNE(n_components=2, perplexity=30, init='random', learning_rate=200)
        print('fitting tsne')
        # Transform the training and test data using t-SNE on top of TSVD 
        X_train_tsne = tsne.fit_transform(X_train_tsvd)
        X_test_tsne = tsne.fit_transform(X_test_tsvd)
        return X_train_tsne , X_test_tsne


class SVM:
    # Initializing the class with some parameters
    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        # Creating an instance of SVC with the given parameters
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    # Fitting the model to the data
    def fit(self, X_train, y_train):
        # Calling the fit method of SVC on the data
        self.model.fit(X_train, y_train)

    # Predicting labels for new data
    def predict(self, X_test):
        # Calling the predict method of SVC on the new data
        return self.model.predict(X_test)

    # Saving the model to a file
    def save(self, filename):
        # Using joblib.dump to serialize the model and save it to a file
        joblib.dump(self.model, filename)

    # Loading the model from a file
    def load(self, filename):
        # Using joblib.load to deserialize and load the model from a file
        self.model = joblib.load(filename)


class RandomForest:
    # Initializing the class with some parameters
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None):
        # Creating an instance of RandomForestClassifier with the given parameters
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    # Fitting the model to the data
    def fit(self, X_train, y_train):
        # Calling the fit method of RandomForestClassifier on the data
        self.model.fit(X_train, y_train)

    # Predicting labels for new data
    def predict(self, X_test):
        # Calling the predict method of RandomForestClassifier on the new data
        return self.model.predict(X_test)

    # Saving the model to a file
    def save(self, filename):
        # Using joblib.dump to serialize the model and save it to a file
        joblib.dump(self.model, filename)

    # Loading the model from a file
    def load(self, filename):
        # Using joblib.load to deserialize and load the model from a file
        self.model = joblib.load(filename)

class DecisionTree:
    def __init__(self,):
        # Initialize the model with optional parameters
        self.model = DecisionTreeClassifier()
        

    def fit(self, X_train, y_train):
        # Train the model on the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Predict the labels for the test data
        return self.model.predict(X_test)

    def save(self, filename):
        # Save the model to a .joblib file
        joblib.dump(self.model, filename)

    def load(self, filename):
        # Load the model from a .joblib file
        self.model = joblib.load(filename)

class KNN:
    # Initializing the class with some parameters
    def __init__(self, n_neighbors=5):
        # Creating an instance of KNeighborsClassifier with the given parameter
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fitting the model to the data
    def fit(self, X_train, y_train):
        # Calling the fit method of KNeighborsClassifier on the data
        self.model.fit(X_train, y_train)

    # Predicting labels for new data
    def predict(self, X_test):
        # Calling the predict method of KNeighborsClassifier on the new data
        return self.model.predict(X_test)

    # Saving the model to a file
    def save(self, filename):
        # Using joblib.dump to serialize the model and save it to a file
        joblib.dump(self.model, filename)

    # Loading the model from a file
    def load(self, filename):
        # Using joblib.load to deserialize and load the model from a file
        self.model = joblib.load(filename)


class AlexNet:
    def __init__(self, HEIGHT, WIDTH, n_outputs):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.n_outputs = n_outputs

    def define_model(self):
        input = Input(shape=(self.HEIGHT, self.WIDTH, 1))

        # first layer
        x = Conv2D(filters=96, kernel_size=11, strides=4,
                   name='conv1', activation='relu')(input)
        x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(2)(x)

        # second layer
        x = Conv2D(filters=256, kernel_size=3, strides=1,
                   name="conv2", activation='relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(1)(x)

        # third layer
        x = Conv2D(filters=384, kernel_size=3, strides=1,
                   name='conv3', activation='relu')(x)
        x = ZeroPadding2D(1)(x)

        # fourth layer
        x = Conv2D(filters=384, kernel_size=3, strides=1,
                   name='conv4', activation='relu')(x)
        x = ZeroPadding2D(1)(x)

        #fifth layer
        x = Conv2D(filters=256, kernel_size=3, strides=1,
                   name='conv5', activation='relu')(x)

        x = Flatten()(x)

        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5, name='dropout_6')(x)

        x = Dense(4096, activation='relu',  name='fc7')(x)
        x = Dropout(0.5, name='dropout_7')(x)

        x = Dense(self.n_outputs, activation='softmax', name='fc8')(x)

        model = Model(inputs=input, outputs=x)
        return model

    def train_model(self, x_train, y_train, x_test, y_test):
        model = self.define_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath='alexnet_model_mnist.h5', monitor='val_loss', save_best_only=True)]
        #train the model
        train = model.fit(x_train, y_train, epochs=20, validation_data=(
            x_test, y_test), callbacks=callbacks)


class Evaluator:
    # Initializing the class with some parameters
    def __init__(self, filename):
        # Loading the model from the joblib file
        self.model = joblib.load(filename)

    # Evaluating the model on some data
    def evaluate(self, X_test, y_test):
        # Predicting labels for the test data using the model
        y_pred = self.model.predict(X_test)

        # Calculating and printing some metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average = 'macro')
        rec = recall_score(y_test, y_pred, average = 'macro')
        f1 = f1_score(y_test, y_pred, average = 'macro')
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1-score: {f1}")
        data = {"Metrics":['Accuracy', 'Precision', 'Recall', 'F1-score'],
                "Values": [acc,prec,rec,f1]}
        df = pd.DataFrame(data = data)
        plt.bar(df["Metrics"],df['Values'])
        plt.title('Evaluation Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        for i , v in enumerate(df['Values']):
            plt.text(i, v, f"{v:.2f}", ha='left', va='center')
        plt.show()
        # Plotting and showing the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    preprocessing = Preprocessing(x_train=x_train, x_test=x_test)
    x_train, x_test = preprocessing.normalization()
    y_preds = {}
    while True:
        print("1.Train and save model Decision Tree")
        print('2.Train and save model Knn ')
        print('3.Train and save model Random Forest')
        print('4.Train and save model SVM ')
        print('5.Train and save model Alexnet')
        choice = int(input("choose your choice: "))
        if choice == 1:
            dt = DecisionTree()
            # Fit the model on the transformed training data
            print('fitting model')
            dt.fit(x_train, y_train)

            
            # Save the model to dt.joblib file
            dt.save("decisiontree.joblib")

            # Load the model from dt.joblib file
            dt.load("decisiontree.joblib")

            # Predict the labels for the transformed test data
            
            y_pred = dt.predict(x_test)
            print(accuracy_score(y_test,y_pred))
        if choice == 2:
            knn = KNN(n_neighbors=3)
            knn.fit(x_train,y_train)

            # Predict labels for the test data
            y_pred = knn.predict(x_test)

            # Save the model to a file named 'knn_model.joblib'
            knn.save('knn_model.joblib')

            # Load the model from a file named 'knn_model.joblib'
            knn.load('knn_model.joblib')
            print(accuracy_score(y_test,y_pred))
        if choice == 3:
            rf = RandomForest(n_estimators=200,
                              criterion="entropy", max_depth=10)
            rf.fit(x_train,y_train)

            # Predict labels for the test data
            y_pred = rf.predict(x_test)

            # Save the model to a file named 'rf_model.joblib'
            rf.save('rf_model.joblib')

            # Load the model from a file named 'rf_model.joblib'
            rf.load('rf_model.joblib')
            print(accuracy_score(y_test,y_pred))
        if choice == 4:
            svm = SVM(kernel="linear", C=0.1 ,gamma=0.01)
            svm.fit(x_train,y_train)

            # Predict labels for the test data
            y_pred = svm.predict(x_test)

            # Save the model to a file named 'svm_model.joblib'
            svm.save('svm_model.joblib')

            # Load the model from a file named 'svm_model.joblib'
            svm.load('svm_model.joblib')
            print(accuracy_score(y_test,y_pred))
        if choice == 5:
            x_train = x_train.reshape(x_train.shape[0], HEIGHT, WIDTH, 1)
            x_test = x_test.reshape(x_test.shape[0], HEIGHT, WIDTH, 1)

            uniques_values, counts = np.unique(y_train, return_counts=True)
            no_classes = len(uniques_values)

            y_train = to_categorical(y_train, no_classes)
            y_test = to_categorical(y_test, no_classes)

            """model = AlexNet(HEIGHT=HEIGHT, WIDTH=WIDTH, n_outputs=no_classes)
            model.train_model(x_train, y_train, x_test, y_test)"""
            model_alexnet = load_model("alexnet_model_mnist.h5")
            y_pred = model_alexnet.predict(x_test)
            y_pred = np.argmax(y_pred, axis = 1)
            y_test = np.argmax(y_test, axis = 1)
            print(accuracy_score(y_test,y_pred))
        if choice == 6:
            file_name = input('select a model: ')
            evaluator = Evaluator(filename = file_name)
            evaluator.evaluate(x_test,y_test)
        else:
            break
