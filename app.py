# Libraries, modules
from flask import Flask, render_template, url_for, request
import json
# Models, utils, helpers, services
from models.Trainers import Trainers
from models.preprocessings import for_dataset as pre_train
from models.Predict_message import Predict_message
from models.Predict_file import Predict_file
app = Flask(__name__)

RESPONSE_DEFAULT = {
    'pre_train_data': None,
    'train_id': '-1',
    'train_result': None,
    'chart_result': None,
    'predict_label': '',
    'predict_result': None
}

res = RESPONSE_DEFAULT

# Router
@app.route('/')
def home():
    # myClass = KNN("spam.csv")
    # myClass.train()
    global res
    res = RESPONSE_DEFAULT.copy()
    return render_template('home.html', res=res)


@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        filename = request.form['customFile']
        # print("filename: {}".format(filename))
        X_train, X_test, y_train, y_test, output = pre_train(filename)
        global res
        res['pre_train_data'] = output
        return render_template('home.html', res=res)
    return render_template('home.html', res=res)


def selectModel(i, option, devDependcy):
    trainer = option(devDependcy)
    switcher = {
        # -1: 'All',
        0: trainer.DecisionTree,
        1: trainer.KNN,
        2: trainer.Naive_bayes,
        3: trainer.LSTM,
        4: trainer.SVM
    }
    return switcher.get(int(i), trainer.Run_All)


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        trainSelected = request.form['model']
        filePath = 'spam.csv'
        print(trainSelected)
        train_result = selectModel(trainSelected, Trainers, filePath)()
        global res
        res['train_id'] = trainSelected
        res['predict_result'] = None
        res['predict_label'] = None
        if int(trainSelected) < 0:
            res['chart_result'] = train_result
            res['train_result'] = None
        else:
            res['train_result'] = train_result.getObj()
            res['chart_result'] = None

        return render_template('home.html', res=res)
    return render_template('home.html', res=res)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        global res
        message = request.form['predictLabel']
        fileName = request.form['filename']
        # fileResponse = request.files['filename']
        trainSelected = res['train_id']
        print(message)
        predict_result = None
        if message != '':
            predict_result = selectModel(
                trainSelected, Predict_message, message)()
        else:
            selectModel(trainSelected, Predict_file, fileName)()
            predict_result = fileName
        res['predict_label'] = message
        res['predict_result'] = predict_result
        return render_template('home.html', res=res)

    return render_template('home.html', res=res)


if __name__ == '__main__':
    app.run(debug=False)
