# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BENCMARK_ITER_N = 1

benchmark_results = pd.DataFrame(
    columns=[
        "model",
        "Data size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "time per data per iter",
    ]
)

def benchmarkAndUpdateResult(X_test, y_test, model, model_name):
    """
    Benchmark the model and update the results dataframe

    Parameters
    ----------
    X_test : The test data
    y_test : The actual test labels
    model : The model to be benchmarked, must have the predict method
    model_name : The name of the model
    """
    global benchmark_results
    data_size = np.shape(X_test)[0]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    iter_n = BENCMARK_ITER_N
    start = time.time()
    for _ in range(iter_n):  # benchmark
        for j in range(data_size):  # simulating real time data
            model.predict(X_test[j : j + 1])
    end = time.time()
    time_per_data_per_iter = (end - start) / data_size / iter_n
    benchmark_results.loc[len(benchmark_results)] = [
        model_name,
        data_size,
        accuracy,
        precision,
        recall,
        f1,
        time_per_data_per_iter,
    ]
    print(f"{model_name} benchmark done")


# %%
# sample binary classification, replace wiht the actual code for the project
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
X = X[y != 0]
y = y[y != 0]
df = pd.DataFrame(X, columns=iris.feature_names)
df["class"] = y
X = df.drop(columns=["class"])
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC()
model.fit(X_train, y_train)

### Add the following line when you are confident in the model you developed
benchmarkAndUpdateResult(X_test, y_test, model, "SVM")

# %%
display(benchmark_results)