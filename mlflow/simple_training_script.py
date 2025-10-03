import mlflow
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

C = 1.0
penalty = "l2"
max_iter = 2000
test_size = 0.2
random_state = 42
solver = "lbfgs"

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

mlflow.set_experiment("iris-logreg")

with mlflow.start_run():
    # Log hyperparameters
    params = {
        "C": C,
        "penalty": penalty,
        "max_iter": max_iter,
        "test_size": test_size,
        "random_state": random_state,
        "solver": solver
    }
    mlflow.log_params(params)

    # Data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                random_state=random_state,
            )),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # Log metrics
    mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})


    # Infer the model signature
    signature = infer_signature(X_train, y_pred)

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="iris-model",
    )

    # Set a tag
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "simple_logreg_iris_model"}
    )

    print(f"Training done: accuracy={acc:.4f} f1_macro={f1:.4f}")

