import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

'''
training code to train the model
'''
def train_model():
    df = pd.read_csv('data/diabetes.csv')
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI',
            'DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    hyperparam_value=0.2
    regression_model = LogisticRegression(C=hyperparam_value, solver="liblinear").fit(X_train, y_train)

    y_hat = regression_model.predict(X_test)
    acc = np.average(y_hat == y_test)

    y_scores = regression_model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])

    print("Accuracy: ",acc)
    # Return relevant information for logging
    return [regression_model,acc,hyperparam_value]
    
if __name__ == "__main__":
    experiment_name="MLOPS Initiative"
    run_name="test_run_"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    mlflow.autolog()

    curr_run_id=""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        # MLflow triggers logging automatically upon model fitting if we are using autolog
        train_model()
        # Also, we can separately log the  metrics and parameters like:
        # 1. Logging Parameters: mlflow.log_param('param', hparam_value)
        # 2. Logging metrics: mlflow.log_metric('accuracy', curr_accuracy)
        # 3. Logging model: mlflow.<framework>.log_model(reg_model, "model")
        # 4. Setting Tags: mlflow.set_tag
        # 5. mlflow.log_artifact

        curr_run_id=run.info.run_id
    
    register_model=input("Do you want to register current model to model registry.\nyes/no\n")
    if (register_model=='yes'):
        model_name="Diabetes_Predictor_test"
        with mlflow.start_run(run_id=curr_run_id) as run:
            result=mlflow.register_model(f"runs:/{curr_run_id}/model",model_name)
        print("Model Registered successfully.")
    else:
        print("Model not registered.")

    # MLFlow provides API for moving the model from staging to production area.
    # We can also achieve the same using MLFlow UI.

