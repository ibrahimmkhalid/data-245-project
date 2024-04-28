try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

# %%
if IN_COLAB:
    # !git clone https://github.com/rapidsai/rapidsai-csp-utils.git
    # !python rapidsai-csp-utils/colab/pip-install.py
    pass

# %%
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.set_option("display.width", 10000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)

TESTING = False
TESTING_SIZE = 0.01
BENCMARK_ITER_N = 10
random_state = 245

benchmark_results = pd.DataFrame(
    columns=[
        "Model",
        "Dataset",
        "Info",
        "Data size",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Time per data per iter",
    ]
)


def benchmarkAndUpdateResult(X_test, y_test, model, model_name, dataset_name, info=""):
    """
    Benchmark the model and update the results dataframe

    Parameters
    ----------
    X_test : The test data
    y_test : The actual test labels
    model : The model to be benchmarked, must have the predict method
    model_name : The name of the model
    dataset_name : The name of the dataset
    info : Additional information about the model
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
        dataset_name,
        info,
        data_size,
        accuracy,
        precision,
        recall,
        f1,
        time_per_data_per_iter,
    ]
    print(f"Model: {model_name}")
    print(f"Data size: {data_size}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Time per data per iter: {time_per_data_per_iter}")


# %%
# sample binary classification, replace wiht the actual code for the project
if IN_COLAB:
    from cuml.svm import SVC
    import cudf
    %load_ext cudf.pandas
else:
    from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# %%
if IN_COLAB:
    prepend_path = "/content/drive/MyDrive/Syncable/sjsu/data-245/DATA 245 Project Files/data"
else:
    prepend_path = "./data"
known_attacks_path = f"{prepend_path}/probe_known_attacks_small.csv"
similar_attacks_path = f"{prepend_path}/probe_similar_attacks_small.csv"
new_attacks_path = f"{prepend_path}/probe_new_attacks_small.csv"

# %%
df = pd.read_csv(known_attacks_path, low_memory=False)

# %%
if TESTING:
    df = df.sample(frac=TESTING_SIZE, random_state=random_state)
    df.reset_index(drop=True, inplace=True)
df.shape

# %%
df.columns

# %%
df.head()

# %%
df.describe(include="all")

# %% [markdown]
# It seems as though ip_RF, ip_MF, and ip_offset do not contain any valuable information. They can be removed

# %%
df = df.drop(columns=["ip_RF", "ip_MF", "ip_offset"])

# %%
df.columns

# %%
print(df["class"].value_counts())
print(df["class"].value_counts(normalize=True) * 100)

# %%
df.dtypes

# %%
df["ip_type"].value_counts()

# %%
df["class"] = df["class"].replace({"normal": 0, "attack": 1})

# %%
df = df.astype(float)

# %%
corr = df.corr()

# %%
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap="Blues")
plt.show()

# %%
corr["class"].sort_values(ascending=False)

# %% [markdown]
# The target feature does not seem to have very strong correlations with any particular feature.

# %%
# remove all features with an absolute correlation of less than 0.1
cols_corr_gt1 = corr["class"][abs(corr["class"]) > 0.1].index

# %%
print(cols_corr_gt1)
print(len(cols_corr_gt1))

# %%
X = df.drop(columns=["class"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(df.columns) - 1)
X_pca = pca.fit_transform(X_scaled)

pca_cumsum = pca.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs Number of components")
plt.grid()
plt.xticks(range(0, len(df.columns) - 1, 2))
plt.show()

# %% [markdown]
# - at n=21, we have 85% of the variance explained
# - at n=24, we have 90% of the variance explained
# - at n=27, we have 95% of the variance explained

# %%
X_gt1 = df[cols_corr_gt1]

scaler_gt1 = StandardScaler()
X_gt1_scaled = scaler_gt1.fit_transform(X_gt1)

pca = PCA(n_components=len(cols_corr_gt1))
X_gt1_pca = pca.fit_transform(X_gt1_scaled)

pca_cumsum = pca.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components with |correlation| > 0.1")
plt.ylabel("Cumulative explained variance")
plt.title(
    "Cumulative explained variance vs Number of components with |correlation| > 0.1"
)
plt.grid()
plt.xticks(range(0, len(cols_corr_gt1), 2))
plt.show()

# %% [markdown]
# - at n=10 of the selected features, we have 95% of the variance explained

# %%
model_params = {
        "C": np.logspace(-3, 4, 8),
        # "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "kernel": ["poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
        }
model_params

# %%
verbose = 0
cv = 3
n_jobs = -1

# %%
df_corr_gt1_scaled = df[cols_corr_gt1[:-1]]
df_corr_gt1_scaler = StandardScaler()
df_corr_gt1_scaled = df_corr_gt1_scaler.fit_transform(df_corr_gt1_scaled)
df_corr_gt1_scaled = pd.DataFrame(
    df_corr_gt1_scaled, columns=cols_corr_gt1[:-1]
)
df_corr_gt1_scaled["class"] = df["class"]
df_corr_gt1_scaled.head()

# %%
df_corr_gt1_scaled = df_corr_gt1_scaled.to_numpy()

# %%
(
    X_corr_gt1_scaled_train,
    X_corr_gt1_scaled_test,
    y_corr_gt1_scaled_train,
    y_corr_gt1_scaled_test,
) = train_test_split(
    df_corr_gt1_scaled[:, :-1],
    df_corr_gt1_scaled[:, -1],
    test_size=0.2,
    random_state=random_state,
)

# %%
model_corr_gt1_scaled = GridSearchCV(SVC(), model_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
model_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
model_corr_gt1_scaled.best_params_

# %%
benchmarkAndUpdateResult(
    X_corr_gt1_scaled_test,
    y_corr_gt1_scaled_test,
    model_corr_gt1_scaled,
    f"SVM {model_corr_gt1_scaled.best_params_}",
    "Known attacks",
    "|correlation| > 0.1 features scaled",
)


# %%
df_full_scaled = df.drop(columns=["class"])
df_full_scaler = StandardScaler()
df_full_scaled = df_full_scaler.fit_transform(df_full_scaled)
df_full_scaled = pd.DataFrame(df_full_scaled, columns=df.drop(columns="class").columns)
df_full_scaled["class"] = df["class"]
df_full_scaled.head()

# %%
df_full_scaled = df_full_scaled.to_numpy()

# %%
X_full_scaled_train, X_full_scaled_test, y_full_scaled_train, y_full_scaled_test = (
    train_test_split(
        df_full_scaled[:, :-1],
        df_full_scaled[:, -1],
        test_size=0.2,
        random_state=random_state,
    )
)

# %%
model_full_scaled = GridSearchCV(SVC(), model_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
model_full_scaled.fit(X_full_scaled_train, y_full_scaled_train)

# %%
model_full_scaled.best_params_

# %%
benchmarkAndUpdateResult(
    X_full_scaled_test,
    y_full_scaled_test,
    model_full_scaled,
    f"SVM {model_full_scaled.best_params_}",
    "Known attacks",
    "All features scaled",
)

# %%
df_full_pca_95 = pd.DataFrame(X_pca[:, :27])
df_full_pca_95["class"] = df["class"]
df_full_pca_95.head()

# %%
df_full_pca_95 = df_full_pca_95.to_numpy()

# %%
X_full_pca_train, X_full_pca_test, y_full_pca_train, y_full_pca_test = train_test_split(
    df_full_pca_95[:, :-1],
    df_full_pca_95[:, -1],
    test_size=0.2,
    random_state=random_state,
)

# %%
model_full_pca = GridSearchCV(SVC(), model_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
model_full_pca.fit(X_full_pca_train, y_full_pca_train)

# %%
model_full_pca.best_params_

# %%
benchmarkAndUpdateResult(
    X_full_pca_test,
    y_full_pca_test,
    model_full_pca,
    f"SVM {model_full_pca.best_params_}",
    "Known attacks",
    "PCA 95% on all features",
)

# %%
df_corr_gt1_pca_95 = pd.DataFrame(X_gt1_pca[:, :10])
df_corr_gt1_pca_95["class"] = df["class"]
df_corr_gt1_pca_95.head()

# %%
df_corr_gt1_pca_95 = df_corr_gt1_pca_95.to_numpy()

# %%
X_corr_gt1_pca_train, X_corr_gt1_pca_test, y_corr_gt1_pca_train, y_corr_gt1_pca_test = (
    train_test_split(
        df_corr_gt1_pca_95[:, :-1],
        df_corr_gt1_pca_95[:, -1],
        test_size=0.2,
        random_state=random_state,
    )
)

# %%
model_corr_gt1_pca = GridSearchCV(SVC(), model_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
model_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
model_corr_gt1_pca.best_params_

# %%
benchmarkAndUpdateResult(
    X_corr_gt1_pca_test,
    y_corr_gt1_pca_test,
    model_corr_gt1_pca,
    f"SVM {model_corr_gt1_pca.best_params_}",
    "Known attacks",
    "PCA 95% on features with |correlation| > 0.1",
)

# %%
benchmark_results

# %% [markdown]
# - Best model in terms of accuracy and time: SVM with features with |correlation| > 0.1 and scaled
# - Testing this model on the similar attacks dataset

# %%
df_similar_attacks = pd.read_csv(similar_attacks_path, low_memory=False)

# %%
df_similar_attacks = df_similar_attacks.drop(columns=["ip_RF", "ip_MF", "ip_offset"])
df_similar_attacks_class = df_similar_attacks["class"].replace(
    {"normal": 0, "attack": 1}
)
df_similar_attacks = df_similar_attacks[cols_corr_gt1[:-1]]
df_similar_attacks_scaled = df_corr_gt1_scaler.transform(
    df_similar_attacks
)
df_similar_attacks_scaled = pd.DataFrame(
    df_similar_attacks_scaled, columns=df_similar_attacks.columns
)
df_similar_attacks_scaled["class"] = df_similar_attacks_class
df_similar_attacks_scaled.head()

# %%
X_similar_attacks = df_similar_attacks_scaled.drop(columns=["class"]).to_numpy()
y_similar_attacks = df_similar_attacks_scaled["class"].to_numpy()

# %%
benchmarkAndUpdateResult(
    X_similar_attacks,
    y_similar_attacks,
    model_corr_gt1_scaled,
    f"SVM {model_corr_gt1_scaled.best_params_}",
    "Similar attacks",
    "|correlation| > 0.1 features scaled",
)

# %%
benchmark_results

# %%
df_new_attacks = pd.read_csv(new_attacks_path, low_memory=False)

# %%
df_new_attacks = df_new_attacks.drop(columns=["ip_RF", "ip_MF", "ip_offset"])
df_new_attacks_class = df_new_attacks["class"].replace({"normal": 0, "attack": 1})
df_new_attacks = df_new_attacks[cols_corr_gt1[:-1]]
df_new_attacks_scaled = df_corr_gt1_scaler.transform(
    df_new_attacks
)
df_new_attacks_scaled = pd.DataFrame(
    df_new_attacks_scaled, columns=df_new_attacks.columns
)
df_new_attacks_scaled["class"] = df_new_attacks_class
df_new_attacks_scaled.head()

# %%
X_new_attacks = df_new_attacks_scaled.drop(columns=["class"]).to_numpy()
y_new_attacks = df_new_attacks_scaled["class"].to_numpy()

# %%
benchmarkAndUpdateResult(
    X_new_attacks,
    y_new_attacks,
    model_corr_gt1_scaled,
    f"SVM {model_corr_gt1_scaled.best_params_}",
    "New attacks",
    "|correlation| > 0.1 features scaled",
)

# %%
benchmark_results
