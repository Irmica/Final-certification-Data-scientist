import flask
from flask import render_template
import pickle
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

app = flask.Flask(__name__, template_folder='src/templates',
                  static_folder='src')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('index.html')

    if flask.request.method == 'POST':
        with open('src/models/mor_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        with open('src/models/norm_params.pkl', 'rb') as np:
            norm_params = pickle.load(np)

        scaler = MinMaxScaler()
        scaler.min_ = norm_params['min']
        scaler.scale_ = norm_params['scale']
        
        input_iw = float(flask.request.form['iw'])
        input_if = float(flask.request.form['if'])
        input_vw = float(flask.request.form['vw'])
        input_fp = float(flask.request.form['fp'])
        
        input_data = [
            [
                input_iw,
                input_if,
                input_vw,
                input_fp
            ]
        ]
        print('\n Введенные данные', input_data, '\n')

        input_data = scaler.transform(input_data)
        print('\n Нормализованные данные', input_data, '\n')
        
        y_pred = loaded_model.predict(input_data)
        print('\n Предсказанные значения', y_pred, '\n')

        context = {
            'iw': input_iw,
            'if': input_if,
            'vw': input_vw,
            'fp': input_fp,
            'depth': round(y_pred[0][0], 2),
            'width': round(y_pred[0][1], 2)
        }

        return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)
