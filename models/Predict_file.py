import pandas as pd
from models import preprocessings
from sklearn.externals import joblib
from tensorflow.keras.models import load_model


def export_file(path, prediction):
    data_test = pd.read_csv(path, encoding="ISO-8859-1")
    data_test['class'] = prediction
    data_test['class'] = data_test['class'].replace(
        [1, 0],  ["spam", "ham"])
    data_test.to_csv(path, index=False)
    print(data_test)


class Predict_file:
    def __init__(self, path):
        self.path = path

    def KNN(self):
        x = preprocessings.for_file(self.path)
        modelscorev2 = joblib.load('KNN.pkl', mmap_mode='r')
        prediction = modelscorev2.predict(x)
        print(prediction)
        export_file(self.path, prediction)

    def DecisionTree(self):
        x = preprocessings.for_file(self.path)
        decisionTree = joblib.load('DecisionTree.pkl', mmap_mode='r')
        prediction = decisionTree.predict(x)
        print(prediction)
        export_file(self.path, prediction)

    def Naive_bayes(self):
        x = preprocessings.for_file(self.path)
        NB = joblib.load('NB.pkl', mmap_mode='r')
        prediction = NB.predict(x)
        print(prediction)
        export_file(self.path, prediction)

    def SVM(self):
        x = preprocessings.for_file(self.path)
        svm = joblib.load('SVM.pkl', mmap_mode='r')
        prediction = svm.predict(x)
        print(prediction)
        export_file(self.path, prediction)

    def LSTM(self):
        x = preprocessings.for_file_lstm(self.path)
        lstm_model = load_model('51_acc_language_model.h5')
        prediction = lstm_model.predict_classes(x)
        print(prediction)
        export_file(self.path, prediction)

    def Run_All(self):
        return None

# p3 = Predict_file("E:\SMS SPAM\Spam-Detection-master\Dataset\Test.csv")
# p3.KNN()
