from flask import Flask, jsonify, request
import numpy as np
import LoanPredictionModel as lpm

app = Flask(__name__)


@app.route('/')
def defaultLoadingContent():
    return 'WeTrack Prediction Website'


@app.route('/LoanPredict', methods=['GET', 'POST'])
def loanApprovalPredict():
    if request.method == "POST":
        req_data = request.get_json()
        val1 = req_data['val1']
        val2 = req_data['val2']
        val3 = req_data['val3']
        val4 = req_data['val4']
        val5 = req_data['val5']

        model = lpm.load_loan_pickle()
        X_test = np.array([[val1, val2, val3, val4, val5]])
        if model != None:
            y_pred = model.predict(X_test)
            return jsonify({"loan_prediction_res": int(y_pred[0]), "loan_predict_status": "Success"})
        else:
            return jsonify({"loan_predict_status": "Failed"})

    else:
        return jsonify({"loan_predict_error": "Call the API using POST Method to predict the loan !!",
                        "loan_predict_status": "Failed"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)