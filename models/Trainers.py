from models import Measure
from models import preprocessings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import svm
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn import metrics, feature_extraction
# from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM as LSTM_lib
from tensorflow.keras.layers import Dense


def DecisionTree(X_train, X_test, y_train, y_test):
    # huấn luyện mô hình bằng tập train và test
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=None)
    dtc.fit(X_train, y_train)
    # dự đoán cho tập dữ liệu test
    y_dtc = dtc.predict(X_test)
    cm = confusion_matrix(y_dtc, y_test)
    print(cm)
    print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # in ra độ chính xác của các label
    print(classification_report(y_test, dtc.predict(X_test)))
    joblib.dump(dtc, 'DecisionTree.pkl')
    precision, recall, fscore, support = score(
        y_test, y_dtc, average='weighted')
    acc_score = accuracy_score(y_test, y_dtc)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def Naive_Bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_mnb = clf.predict(X_test)
    cm = confusion_matrix(y_mnb, y_test)
    print(cm)
    print('Naive Bayes Accuracy: ', accuracy_score(y_test, y_mnb))
    # #in ra độ chính xác của các label
    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, 'NB.pkl')
    precision, recall, fscore, support = score(
        y_test, y_mnb, average='weighted')
    acc_score = accuracy_score(y_test, y_mnb)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def KNN(X_train, X_test, y_train, y_test):
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train, y_train)
    y_knc = KNN.predict(X_test)
    print('KNeighbors Accuracy_score: ', accuracy_score(y_test, y_knc))
    print('KNeighbors confusion_matrix:/n', confusion_matrix(y_test, y_knc))
    print(classification_report(y_test, KNN.predict(X_test)))
    joblib.dump(KNN, 'KNN.pkl')
    precision, recall, fscore, support = score(
        y_test, y_knc, average='weighted')
    acc_score = accuracy_score(y_test, y_knc)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def SVM(X_train, X_test, y_train, y_test):
    SVM = svm.SVC(kernel='linear')  # Linear Kernel
    # Train the model using the training sets
    SVM.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = SVM.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)
    print('svm  Accuracy: ', accuracy_score(y_test, y_pred))
    # #in ra độ chính xác của các label
    print(classification_report(y_test, SVM.predict(X_test)))
    joblib.dump(SVM, 'SVM.pkl')
    precision, recall, fscore, support = score(
        y_test, y_pred, average='weighted')
    acc_score = accuracy_score(y_test, y_pred)
    # print('Decision Tree Accuracy: ', accuracy_score(y_test, y_dtc))
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return Measure.Measure(acc_score, precision, recall, fscore)


def LSTM(X_train_seq, X_test_seq, y_train, y_test):
    max_words = 10000
    max_len = 200
    lstm_model = Sequential()
    lstm_model.add(Embedding(max_words, 50, input_length=max_len))
    lstm_model.add(LSTM_lib(128, dropout=0.25, recurrent_dropout=0.25))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    print('Train model')
    lstm_model.fit(X_train_seq, y_train,
              batch_size=32,
              epochs=3,
              validation_data=(X_test_seq, y_test))
    y_pred = lstm_model.predict_classes(X_test_seq)
    lstm_model.save('51_acc_language_model.h5')
    acc_score = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = score(
        y_test, y_pred, average='weighted')
    target_names = ['Non-Spam', 'Spam']
    print(classification_report(y_test, y_pred, target_names=target_names))
    return Measure.Measure(acc_score, precision, recall, fscore)


class Trainers:
    def __init__(self, path):
        self.path = path

    def KNN(self):
        X_train, X_test, y_train, y_test, output = preprocessings.for_dataset(
            self.path)
        return KNN(X_train, X_test, y_train, y_test)

    def DecisionTree(self):
        X_train, X_test, y_train, y_test, output = preprocessings.for_dataset(
            self.path)
        return DecisionTree(X_train, X_test, y_train, y_test)

    def Naive_bayes(self):
        X_train, X_test, y_train, y_test, output = preprocessings.for_dataset(
            self.path)
        return Naive_Bayes(X_train, X_test, y_train, y_test)

    def SVM(self):
        X_train, X_test, y_train, y_test, output = preprocessings.for_dataset(
            self.path)
        return SVM(X_train, X_test, y_train, y_test)

    def LSTM(self):
        X_train_seq, X_test_seq, y_train, y_test = preprocessings.for_dataset_lstm(
            self.path)
        return LSTM(X_train_seq, X_test_seq, y_train, y_test)

    def Run_All(self):
        return [
            {
                'trainer': 'KNN',
                'result': self.KNN().getObj()
            },
            {
                'trainer': 'DecisionTree',
                'result': self.DecisionTree().getObj()
            },
            {
                'trainer': 'Naive_bayes',
                'result': self.Naive_bayes().getObj()
            },
            {
                'trainer': 'SVM',
                'result': self.SVM().getObj()
            },
            {
                'trainer': 'LSTM',
                'result': self.LSTM().getObj()
            },
        ]


# p1 = Trainers("spam.csv")
# p1=SMS_spam_detect()

# p1.SVM_predict("I'm gonna be home soon and i don't want to talk about this https://stackoverflow.com/questions/44193154/notfittederror-tfidfvectorizer-vocabulary-wasnt-fitted stuff anymore tonight, k? I've cried enough today. ")
# p1.LSTM()
