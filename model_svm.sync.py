try:
    from google.colab import drive

    drive.mount("/content/drive")
    IN_COLAB = True
except:
    IN_COLAB = False

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import time

# %%
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


def benchmarkAndUpdateResult(df, model, model_name, dataset_name, info, pipeline_fn, **pipeline_kwargs):
    global benchmark_results
    df_ = pipeline_fn(df=df, **pipeline_kwargs)
    X = df_[df_.columns[:-1]]
    y = df_[df_.columns[-1]]
    data_size = np.shape(X)[0]
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    iter_n = BENCMARK_ITER_N
    start = time.perf_counter_ns()
    for _ in range(iter_n):  # benchmark
        df_ = pipeline_fn(df=df, **pipeline_kwargs)
        X = df_[df_.columns[:-1]]
        model.predict(X)
    end = time.perf_counter_ns()
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
    print(classification_report(y, y_pred))
    print()
    print(f"Model: {model_name}")
    print(f"Data size: {data_size}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Time per data per iter: {time_per_data_per_iter}")


# %%
def test_train_val_split(df, random_state=random_state):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# %%
if IN_COLAB:
    prepend_path = (
        "/content/drive/MyDrive/Syncable/sjsu/data-245/DATA 245 Project Files/data"
    )
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
# - at n=27, we have 95% of the variance explained

# %%
X_gt1 = df[cols_corr_gt1]
X_gt1 = X_gt1.drop(columns=["class"])

scaler_gt1 = StandardScaler()
X_gt1_scaled = scaler_gt1.fit_transform(X_gt1)

pca_corr_gt1 = PCA(n_components=len(cols_corr_gt1) - 1)
X_gt1_pca = pca_corr_gt1.fit_transform(X_gt1_scaled)

pca_cumsum = pca_corr_gt1.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components with |correlation| > 0.1")
plt.ylabel("Cumulative explained variance")
plt.title(
    "Cumulative explained variance vs Number of components with |correlation| > 0.1"
)
plt.grid()
plt.xticks(range(0, len(cols_corr_gt1) - 1, 2))
plt.show()

# %% [markdown]
# - at n=10 of the selected features, we have 95% of the variance explained

# %% [markdown]
# # Modelling

# %%
_, _, __X__, _, _, __y__ = test_train_val_split(df)
df_known_attacks = pd.DataFrame(__X__)
df_known_attacks["class"] = __y__
df_known_attacks.reset_index(drop=True, inplace=True)

# %%
df_similar_attacks = pd.read_csv(similar_attacks_path, low_memory=False)
df_similar_attacks = df_similar_attacks.drop(columns=["ip_RF", "ip_MF", "ip_offset"])
df_similar_attacks["class"] = df_similar_attacks["class"].replace({"normal": 0, "attack": 1})

df_new_attacks = pd.read_csv(new_attacks_path, low_memory=False)
df_new_attacks = df_new_attacks.drop(columns=["ip_RF", "ip_MF", "ip_offset"])
df_new_attacks["class"] = df_new_attacks["class"].replace({"normal": 0, "attack": 1})

# %%
if TESTING:
    df_similar_attacks = df_similar_attacks.sample(frac=TESTING_SIZE, random_state=random_state)
    df_similar_attacks.reset_index(drop=True, inplace=True)
    df_new_attacks = df_new_attacks.sample(frac=TESTING_SIZE, random_state=random_state)
    df_new_attacks.reset_index(drop=True, inplace=True)

# %%
def pipeline_scaled(**kwargs):
    if "df" not in kwargs or "scaler" not in kwargs:
        raise ValueError("df and scaler must be passed as keyword arguments for pipeline_scaled")
    df = kwargs["df"]
    scaler = kwargs["scaler"]

    df_ = df.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pd.DataFrame(df_, columns=df.columns[:-1])
    df_["class"] = df["class"]
    return df_

# %%
def pipeline_corr_gt1_scaled(**kwargs):
    if "df" not in kwargs or "scaler" not in kwargs or "cols" not in kwargs:
        raise ValueError("df, scaler, and cols must be passed as keyword arguments for pipeline_corr_gt1_scaled")
    df = kwargs["df"]
    scaler = kwargs["scaler"]
    cols = kwargs["cols"]

    df_ = df[cols]
    df_ = df_.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pd.DataFrame(df_, columns=cols[:-1])
    df_["class"] = df["class"]
    return df_

# %%
def pipeline_pca(**kwargs):
    if "df" not in kwargs or "scaler" not in kwargs or "pca" not in kwargs:
        raise ValueError("df, scaler, and pca must be passed as keyword arguments for pipeline_pca")
    df = kwargs["df"]
    scaler = kwargs["scaler"]
    pca = kwargs["pca"]

    df_ = df.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pca.transform(df_)
    df_ = df_[:, :27]
    df_ = pd.DataFrame(df_)
    df_["class"] = df["class"]
    return df_

# %%
def pipeline_corr_gt1_pca(**kwargs):
    if "df" not in kwargs or "scaler" not in kwargs or "cols" not in kwargs or "pca" not in kwargs:
        raise ValueError("df, scaler, cols, and pca must be passed as keyword arguments for pipeline_corr_gt1_pca")
    df = kwargs["df"]
    scaler = kwargs["scaler"]
    cols = kwargs["cols"]
    pca = kwargs["pca"]

    df_ = df[cols]
    df_ = df_.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pca.transform(df_)
    df_ = df_[:, :10]
    df_ = pd.DataFrame(df_)
    df_["class"] = df["class"]
    return df_


# %% [markdown]
# ## SVM

# %%
svm_params = {
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["poly", "rbf", "sigmoid", "linear"],
    "gamma": ["scale", "auto"],
}

# %%
verbose = 3
cv = 3
n_jobs = None


# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler)
df_scaled.head()

# %%
(
    X_scaled_train,
    X_scaled_val,
    X_scaled_test,
    y_scaled_train,
    y_scaled_val,
    y_scaled_test,
) = test_train_val_split(df_scaled)

# %%
svm_scaled_baseline = SVC(random_state=random_state)
svm_scaled_baseline.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_val, svm_scaled_baseline.predict(X_scaled_val)))

# %%
svm_scaled_grid = GridSearchCV(SVC(random_state=random_state), svm_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
svm_scaled_grid.fit(X_scaled_val, y_scaled_val)

# %%
print(svm_scaled_grid.best_params_)

# %%
svm_scaled = SVC(**svm_scaled_grid.best_params_, random_state=random_state)
svm_scaled.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_test, svm_scaled.predict(X_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        svm_scaled,
        f"SVM {svm_scaled_grid.best_params_}",
        "Known attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_scaled,
        f"SVM {svm_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_scaled,
        f"SVM {svm_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler
        )
# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_gt1, cols=cols_corr_gt1)
df_corr_gt1_scaled.head()

# %%
(
    X_corr_gt1_scaled_train,
    X_corr_gt1_scaled_val,
    X_corr_gt1_scaled_test,
    y_corr_gt1_scaled_train,
    y_corr_gt1_scaled_val,
    y_corr_gt1_scaled_test,
) = test_train_val_split(df_corr_gt1_scaled)

# %%
svm_corr_gt1_scaled_baseline = SVC(random_state=random_state)
svm_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_val, svm_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))

# %%
svm_corr_gt1_scaled_grid = GridSearchCV(SVC(random_state=random_state), svm_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
svm_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)

# %%
print(svm_corr_gt1_scaled_grid.best_params_)

# %%
svm_corr_gt1_scaled = SVC(**svm_corr_gt1_scaled_grid.best_params_, random_state=random_state)
svm_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_test, svm_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        svm_corr_gt1_scaled,
        f"SVM {svm_corr_gt1_scaled_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_corr_gt1_scaled,
        f"SVM {svm_corr_gt1_scaled_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_gt1,
        cols=cols_corr_gt1
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_corr_gt1_scaled,
        f"SVM {svm_corr_gt1_scaled_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_gt1,
        cols=cols_corr_gt1
        )
# %%
benchmark_results

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler, pca=pca)
df_pca.head()

# %%
(
    X_pca_train,
    X_pca_val,
    X_pca_test,
    y_pca_train,
    y_pca_val,
    y_pca_test,
) = test_train_val_split(df_pca)

# %%
svm_pca_baseline = SVC(random_state=random_state)
svm_pca_baseline.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_val, svm_pca_baseline.predict(X_pca_val)))

# %%
svm_pca_grid = GridSearchCV(SVC(random_state=random_state), svm_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
svm_pca_grid.fit(X_pca_val, y_pca_val)

# %%
print(svm_pca_grid.best_params_)

# %%
svm_pca = SVC(**svm_pca_grid.best_params_, random_state=random_state)
svm_pca.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_test, svm_pca.predict(X_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        svm_pca,
        f"SVM {svm_pca_grid.best_params_}",
        "Known attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler,
        pca=pca
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_pca,
        f"SVM {svm_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler,
        pca=pca
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_pca,
        f"SVM {svm_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler,
        pca=pca
        )
# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1)
df_corr_gt1_pca.head()

# %%
(
    X_corr_gt1_pca_train,
    X_corr_gt1_pca_val,
    X_corr_gt1_pca_test,
    y_corr_gt1_pca_train,
    y_corr_gt1_pca_val,
    y_corr_gt1_pca_test,
) = test_train_val_split(df_corr_gt1_pca)

# %%
svm_corr_gt1_pca_baseline = SVC(random_state=random_state)
svm_corr_gt1_pca_baseline.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_val, svm_corr_gt1_pca_baseline.predict(X_corr_gt1_pca_val)))

# %%
svm_corr_gt1_pca_grid = GridSearchCV(SVC(random_state=random_state), svm_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
svm_corr_gt1_pca_grid.fit(X_corr_gt1_pca_val, y_corr_gt1_pca_val)

# %%
print(svm_corr_gt1_pca_grid.best_params_)

# %%
svm_corr_gt1_pca = SVC(**svm_corr_gt1_pca_grid.best_params_, random_state=random_state)
svm_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_test, svm_corr_gt1_pca.predict(X_corr_gt1_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        svm_corr_gt1_pca,
        f"SVM {svm_corr_gt1_pca_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_corr_gt1_pca,
        f"SVM {svm_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_corr_gt1_pca,
        f"SVM {svm_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1
        )
# %%
benchmark_results
