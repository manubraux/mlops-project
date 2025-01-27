import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
import pickle

def data_serializer(data):
    return {
        "X_train": data["X_train"].tolist(),
        "X_test": data["X_test"].tolist(),
        "y_train": data["y_train"].tolist(),
        "y_test": data["y_test"].tolist()
    }

def data_deserializer(data):
    return {
        "X_train": np.array(data["X_train"]),
        "X_test": np.array(data["X_test"]),
        "y_train": np.array(data["y_train"]),
        "y_test": np.array(data["y_test"])
    }

def model_serializer(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

def model_deserializer():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

with DAG(
    "dag_imt_example_with_mlflow",
    # [START default_args]
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={"retries": 2},
    # [END default_args]
    description="DAG example for the MLOps cours 2025",
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:
    data = {
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None
    }

    model = LinearRegression()
    
    def get_data(**kwargs):
        ti = kwargs["ti"]
        np.random.seed(42)
        X = np.random.rand(100, 1)
        y = 3.5 * X.squeeze() + np.random.randn(100) * 0.5
        X_train_, X_test_, y_train_, y_test_ = train_test_split(
            X, y, test_size=0.2, random_state=42)
        data["X_train"] = X_train_
        data["X_test"] = X_test_
        data["y_train"] = y_train_
        data["y_test"] = y_test_
        ti.xcom_push("data", data_serializer(data))


    def train_model(**kwargs):
        mlflow.autolog()
        data = data_deserializer(kwargs["ti"].xcom_pull(task_ids="get_data", key="data"))
        print(f"{data}")
        model.fit(data["X_train"], data["y_train"])
        model_serializer(model)
        mlflow.sklearn.log_model(model, "model")

    def validate_model(**kwargs):
        data = data_deserializer(kwargs["ti"].xcom_pull(task_ids="get_data", key="data"))
        model = model_deserializer()    
        score = model.score(data["X_test"], data["y_test"])
        print(f"Model score: {score}")

    # Task definition

    get_data_task = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    validate_model_task = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    get_data_task >> train_model_task >> validate_model_task
