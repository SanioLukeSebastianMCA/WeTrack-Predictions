import pandas as pd
from sklearn.naive_bayes import GaussianNB
from os.path import isfile, dirname, join
import pickle


def loan_prediction_model():
    c_file = join(dirname(__file__), "cibil_score_dataset.csv")
    cibil_data = pd.read_csv(c_file)
    X = cibil_data.iloc[:, :-1].values
    y = cibil_data.iloc[:, -1].values
    gnb = GaussianNB()
    gnb.fit(X, y.ravel())
    with open('loan_pred_pickle_model.pkl', 'wb') as f:
        pickle.dump(gnb, f)
    return pickle.load(open("loan_pred_pickle_model.pkl", "rb"))


def load_loan_pickle():
    model_path = "loan_pred_pickle_model.pkl"
    if isfile(model_path):
        with open(model_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                return None
    else:
        return loan_prediction_model()
