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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import time

# %%
pd.set_option("display.width", 10000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)

TESTING = True
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

scaler_standard = StandardScaler()
X_scaled = scaler_standard.fit_transform(X)

pca_standard = PCA(n_components=len(df.columns) - 1)
X_pca = pca_standard.fit_transform(X_scaled)

pca_cumsum = pca_standard.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs Number of components\nUsing Standard Scaler")
plt.grid()
plt.xticks(range(0, len(df.columns) - 1, 2))
plt.show()

# %%
print(f"at n={np.where(pca_cumsum > 0.95)[0][0]}, we have 95% of the variance explained")

# %%
X_gt1 = df[cols_corr_gt1]
X_gt1 = X_gt1.drop(columns=["class"])

scaler_standard_gt1 = StandardScaler()
X_gt1_scaled = scaler_standard_gt1.fit_transform(X_gt1)

pca_corr_gt1_standard = PCA(n_components=len(cols_corr_gt1) - 1)
X_gt1_pca = pca_corr_gt1_standard.fit_transform(X_gt1_scaled)

pca_cumsum = pca_corr_gt1_standard.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components with |correlation| > 0.1")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs Number of components with |correlation| > 0.1\nUsing Standard Scaler")
plt.grid()
plt.xticks(range(0, len(cols_corr_gt1) - 1, 2))
plt.show()

# %%
print(f"at n={np.where(pca_cumsum > 0.95)[0][0]}, we have 95% of the variance explained")

# # %%
# scaler_minmax = MinMaxScaler()
# X_scaled = scaler_minmax.fit_transform(X)
#
# pca_minmax= PCA(n_components=len(df.columns) - 1)
# X_pca = pca_minmax.fit_transform(X_scaled)
#
# pca_cumsum = pca_minmax.explained_variance_ratio_.cumsum()
# plt.plot(pca_cumsum)
# plt.xlabel("Number of components")
# plt.ylabel("Cumulative explained variance")
# plt.title("Cumulative explained variance vs Number of components\nUsing MinMax Scaler")
# plt.grid()
# plt.xticks(range(0, len(df.columns) - 1, 2))
# plt.show()
#
# # %%
# print(f"at n={np.where(pca_cumsum > 0.95)[0][0]}, we have 95% of the variance explained")
#
# # %%
# scaler_minmax_gt1 = MinMaxScaler()
# X_gt1_scaled = scaler_minmax_gt1.fit_transform(X_gt1)
#
# pca_corr_gt1_minmax = PCA(n_components=len(cols_corr_gt1) - 1)
# X_gt1_pca = pca_corr_gt1_minmax.fit_transform(X_gt1_scaled)
#
# pca_cumsum = pca_corr_gt1_minmax.explained_variance_ratio_.cumsum()
# plt.plot(pca_cumsum)
# plt.xlabel("Number of components with |correlation| > 0.1")
# plt.ylabel("Cumulative explained variance")
# plt.title("Cumulative explained variance vs Number of components with |correlation| > 0.1\nUsing MinMax Scaler")
# plt.grid()
# plt.xticks(range(0, len(cols_corr_gt1) - 1, 2))
# plt.show()
#
# # %%
# print(f"at n={np.where(pca_cumsum > 0.95)[0][0]}, we have 95% of the variance explained")

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
    pca_cols = np.where(pca.explained_variance_ratio_.cumsum() > 0.95)[0][0]

    df_ = df.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pca.transform(df_)
    df_ = df_[:, :pca_cols]
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
    pca_cols = np.where(pca.explained_variance_ratio_.cumsum() > 0.95)[0][0]


    df_ = df[cols]
    df_ = df_.drop(columns=["class"])
    df_ = scaler.transform(df_)
    df_ = pca.transform(df_)
    df_ = df_[:, :pca_cols]
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
verbose = 2
cv = 3
n_jobs = None


# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler_standard)
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
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_scaled,
        f"SVM {svm_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_scaled,
        f"SVM {svm_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1)
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
        scaler=scaler_standard_gt1,
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
        scaler=scaler_standard_gt1,
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
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler_standard, pca=pca_standard)
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
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_pca,
        f"SVM {svm_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_pca,
        f"SVM {svm_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1_standard)
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
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        svm_corr_gt1_pca,
        f"SVM {svm_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )
# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        svm_corr_gt1_pca,
        f"SVM {svm_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
# # %% [markdown]
# # ## Complement Naive Bayes
#
# # %%
# cnb_params = {
#     "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     "force_alpha": [True, False],
#     "fit_prior": [True, False],
#     "norm": [True, False],
# }
#
# # %% [markdown]
# # ### All features scaled
#
# # %%
# df_scaled = pipeline_scaled(df=df, scaler=scaler_minmax)
# df_scaled.head()
#
# # %%
# (
#     X_scaled_train,
#     X_scaled_val,
#     X_scaled_test,
#     y_scaled_train,
#     y_scaled_val,
#     y_scaled_test,
# ) = test_train_val_split(df_scaled)
#
# # %%
# cnb_scaled_baseline = ComplementNB()
# cnb_scaled_baseline.fit(X_scaled_train, y_scaled_train)
#
# # %%
# print(classification_report(y_scaled_val, cnb_scaled_baseline.predict(X_scaled_val)))
#
# # %%
# cnb_scaled_grid = GridSearchCV(ComplementNB(), cnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
# cnb_scaled_grid.fit(X_scaled_val, y_scaled_val)
#
# # %%
# print(cnb_scaled_grid.best_params_)
#
# # %%
# cnb_scaled = ComplementNB(**cnb_scaled_grid.best_params_)
# cnb_scaled.fit(X_scaled_train, y_scaled_train)
#
# # %%
# print(classification_report(y_scaled_test, cnb_scaled.predict(X_scaled_test)))
#
# # %%
# benchmarkAndUpdateResult(
#         df_known_attacks,
#         cnb_scaled,
#         f"ComplementNB {cnb_scaled_grid.best_params_}",
#         "Known attacks",
#         "All features scaled",
#         pipeline_scaled,
#         scaler=scaler_minmax
#         )
#
# # %%
# benchmarkAndUpdateResult(
#         df_similar_attacks,
#         cnb_scaled,
#         f"ComplementNB {cnb_scaled_grid.best_params_}",
#         "Similar attacks",
#         "All features scaled",
#         pipeline_scaled,
#         scaler=scaler_minmax
#         )
#
# # %%
# benchmarkAndUpdateResult(
#         df_new_attacks,
#         cnb_scaled,
#         f"ComplementNB {cnb_scaled_grid.best_params_}",
#         "New attacks",
#         "All features scaled",
#         pipeline_scaled,
#         scaler=scaler_minmax
#         )
#
# # %% [markdown]
# # ### Features with |correlation| > 0.1 scaled
#
# # %%
# df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_minmax_gt1, cols=cols_corr_gt1)
# df_corr_gt1_scaled.head()
#
# # %%
# (
#     X_corr_gt1_scaled_train,
#     X_corr_gt1_scaled_val,
#     X_corr_gt1_scaled_test,
#     y_corr_gt1_scaled_train,
#     y_corr_gt1_scaled_val,
#     y_corr_gt1_scaled_test,
# ) = test_train_val_split(df_corr_gt1_scaled)
#
# # %%
# cnb_corr_gt1_scaled_baseline = ComplementNB()
# cnb_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)
#
# # %%
# print(classification_report(y_corr_gt1_scaled_val, cnb_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))
#
# # %%
# cnb_corr_gt1_scaled_grid = GridSearchCV(ComplementNB(), cnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
# cnb_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)
#
# # %%
# print(cnb_corr_gt1_scaled_grid.best_params_)
#
# # %%
# cnb_corr_gt1_scaled = ComplementNB(**cnb_corr_gt1_scaled_grid.best_params_)
# cnb_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)
#
# # %%
# print(classification_report(y_corr_gt1_scaled_test, cnb_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))
#
# # %%
# benchmarkAndUpdateResult(
#         df_known_attacks,
#         cnb_corr_gt1_scaled,
#         f"ComplementNB {cnb_corr_gt1_scaled_grid.best_params_}",
#         "Known attacks",
#         "|correlation| > 0.1 features scaled",
#         pipeline_corr_gt1_scaled,
#         scaler=scaler_minmax_gt1,
#         cols=cols_corr_gt1
#         )
#
# # %%
# benchmarkAndUpdateResult(
#         df_similar_attacks,
#         cnb_corr_gt1_scaled,
#         f"ComplementNB {cnb_corr_gt1_scaled_grid.best_params_}",
#         "Similar attacks",
#         "|correlation| > 0.1 features scaled",
#         pipeline_corr_gt1_scaled,
#         scaler=scaler_minmax_gt1,
#         cols=cols_corr_gt1
#         )
#
# # %%
# benchmarkAndUpdateResult(
#         df_new_attacks,
#         cnb_corr_gt1_scaled,
#         f"ComplementNB {cnb_corr_gt1_scaled_grid.best_params_}",
#         "New attacks",
#         "|correlation| > 0.1 features scaled",
#         pipeline_corr_gt1_scaled,
#         scaler=scaler_minmax_gt1,
#         cols=cols_corr_gt1
#         )
#
# %% [markdown]
# ## Gaussian Naive Bayes

# %%
gnb_params = {'var_smoothing': np.logspace(0,-9, num=100)}

# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler_standard)
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
gnb_scaled_baseline = GaussianNB()
gnb_scaled_baseline.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_val, gnb_scaled_baseline.predict(X_scaled_val)))

# %%
gnb_scaled_grid = GridSearchCV(GaussianNB(), gnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
gnb_scaled_grid.fit(X_scaled_val, y_scaled_val)

# %%
print(gnb_scaled_grid.best_params_)

# %%
gnb_scaled = GaussianNB(**gnb_scaled_grid.best_params_)
gnb_scaled.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_test, gnb_scaled.predict(X_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        gnb_scaled,
        f"GaussianNB {gnb_scaled_grid.best_params_}",
        "Known attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        gnb_scaled,
        f"GaussianNB {gnb_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        gnb_scaled,
        f"GaussianNB {gnb_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1)
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
gnb_corr_gt1_scaled_baseline = GaussianNB()
gnb_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_val, gnb_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))

# %%
gnb_corr_gt1_scaled_grid = GridSearchCV(GaussianNB(), gnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
gnb_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)

# %%
print(gnb_corr_gt1_scaled_grid.best_params_)

# %%
gnb_corr_gt1_scaled = GaussianNB(**gnb_corr_gt1_scaled_grid.best_params_)
gnb_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_test, gnb_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        gnb_corr_gt1_scaled,
        f"GaussianNB {gnb_corr_gt1_scaled_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        gnb_corr_gt1_scaled,
        f"GaussianNB {gnb_corr_gt1_scaled_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        gnb_corr_gt1_scaled,
        f"GaussianNB {gnb_corr_gt1_scaled_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler_standard, pca=pca_standard)
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
gnb_pca_baseline = GaussianNB()
gnb_pca_baseline.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_val, gnb_pca_baseline.predict(X_pca_val)))

# %%
gnb_pca_grid = GridSearchCV(GaussianNB(), gnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
gnb_pca_grid.fit(X_pca_val, y_pca_val)

# %%
print(gnb_pca_grid.best_params_)

# %%
gnb_pca = GaussianNB(**gnb_pca_grid.best_params_)
gnb_pca.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_test, gnb_pca.predict(X_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        gnb_pca,
        f"GaussianNB {gnb_pca_grid.best_params_}",
        "Known attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        gnb_pca,
        f"GaussianNB {gnb_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        gnb_pca,
        f"GaussianNB {gnb_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1_standard)
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
gnb_corr_gt1_pca_baseline = GaussianNB()
gnb_corr_gt1_pca_baseline.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_val, gnb_corr_gt1_pca_baseline.predict(X_corr_gt1_pca_val)))

# %%
gnb_corr_gt1_pca_grid = GridSearchCV(GaussianNB(), gnb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
gnb_corr_gt1_pca_grid.fit(X_corr_gt1_pca_val, y_corr_gt1_pca_val)

# %%
print(gnb_corr_gt1_pca_grid.best_params_)

# %%
gnb_corr_gt1_pca = GaussianNB(**gnb_corr_gt1_pca_grid.best_params_)
gnb_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_test, gnb_corr_gt1_pca.predict(X_corr_gt1_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        gnb_corr_gt1_pca,
        f"GaussianNB {gnb_corr_gt1_pca_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        gnb_corr_gt1_pca,
        f"GaussianNB {gnb_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        gnb_corr_gt1_pca,
        f"GaussianNB {gnb_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmark_results

# %% [markdown]
# ## Logistic Regression

# %%
log_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem
    'penalty': ['l1', 'l2', 'elasticnet', 'none']  # Norm used in the penalization
}

# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler_standard)
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
log_scaled_baseline = LogisticRegression(random_state=random_state)
log_scaled_baseline.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_val, log_scaled_baseline.predict(X_scaled_val)))

# %%
log_scaled_grid = GridSearchCV(LogisticRegression(random_state=245), log_param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
log_scaled_grid.fit(X_scaled_val, y_scaled_val)

# %%
print(log_scaled_grid.best_params_)

# %%
log_scaled = LogisticRegression(**log_scaled_grid.best_params_, random_state=random_state)
log_scaled.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_test, log_scaled.predict(X_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        log_scaled,
        f"Logistic Regression {log_scaled_grid.best_params_}",
        "Known attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        log_scaled,
        f"Logistic Regression {log_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        log_scaled,
        f"Logistic Regression {log_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1)
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
log_corr_gt1_scaled_baseline = LogisticRegression(random_state=random_state)
log_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_val, log_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))

# %%
log_corr_gt1_scaled_grid = GridSearchCV(LogisticRegression(random_state=random_state), log_param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
log_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)

# %%
print(log_corr_gt1_scaled_grid.best_params_)

# %%
log_corr_gt1_scaled = LogisticRegression(**log_corr_gt1_scaled_grid.best_params_, random_state=random_state)
log_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_test, log_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        log_corr_gt1_scaled,
        f"Logistic Regression {log_corr_gt1_scaled_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        log_corr_gt1_scaled,
        f"Logistic Regression {log_corr_gt1_scaled_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        log_corr_gt1_scaled,
        f"Logistic Regression{log_corr_gt1_scaled_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmark_results

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler_standard, pca=pca_standard)
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
log_pca_baseline = LogisticRegression(random_state=random_state)
log_pca_baseline.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_val, log_pca_baseline.predict(X_pca_val)))

# %%
log_pca_grid = GridSearchCV(LogisticRegression(random_state=random_state), log_param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
log_pca_grid.fit(X_pca_val, y_pca_val)

# %%
print(log_pca_grid.best_params_)

# %%
log_pca = LogisticRegression(**log_pca_grid.best_params_, random_state=random_state)
log_pca.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_test, log_pca.predict(X_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        log_pca,
        f"Logistic Regression {log_pca_grid.best_params_}",
        "Known attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        log_pca,
        f"Logistic Regressiion {log_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        log_pca,
        f"Logistic Regression {log_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1_standard)
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
log_corr_gt1_pca_baseline = LogisticRegression(random_state=random_state)
log_corr_gt1_pca_baseline.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_val, log_corr_gt1_pca_baseline.predict(X_corr_gt1_pca_val)))

# %%
log_corr_gt1_pca_grid = GridSearchCV(LogisticRegression(random_state=random_state), log_param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
log_corr_gt1_pca_grid.fit(X_corr_gt1_pca_val, y_corr_gt1_pca_val)

# %%
print(log_corr_gt1_pca_grid.best_params_)

# %%
log_corr_gt1_pca = LogisticRegression(**log_corr_gt1_pca_grid.best_params_, random_state=random_state)
log_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_test, log_corr_gt1_pca.predict(X_corr_gt1_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        log_corr_gt1_pca,
        f"Logisitc Regression {log_corr_gt1_pca_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        log_corr_gt1_pca,
        f"Logistic Regression {log_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        log_corr_gt1_pca,
        f"Logistic Regression {log_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmark_results
# %% [markdown]
# ## XGBoost

# %%
xgb_params = {
    'max_depth': [10, 20, 30, 40, 50],
    'n_estimators': [100, 200, 400, 800],
    'learning_rate': [0.01, 0.1, 0.2],
    'gamma': [0.1, 0.5, 1],
    'scale_pos_weight': [0.1, 1, 5, 10]
}

# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler_standard)
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
xgb_mod_base = XGBClassifier(random_state=random_state)
xgb_mod_base.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_val, xgb_mod_base.predict(X_scaled_val)))

# %%
xgb_scaled_grid = GridSearchCV(XGBClassifier(random_state=random_state), xgb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
xgb_scaled_grid.fit(X_scaled_val, y_scaled_val)

# %%
print(xgb_scaled_grid.best_params_)

# %%
xgb_scaled = XGBClassifier(**xgb_scaled_grid.best_params_, random_state=random_state)
xgb_scaled.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_test, xgb_scaled.predict(X_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        xgb_scaled,
        f"XGBClassifier {xgb_scaled_grid.best_params_}",
        "Known attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        xgb_scaled,
        f"XGBClassifier {xgb_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        xgb_scaled,
        f"XGBClassifier {xgb_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1)
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
xgb_corr_gt1_scaled_baseline = XGBClassifier(random_state=random_state)
xgb_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_val, xgb_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))

# %%
xgb_corr_gt1_scaled_grid = GridSearchCV(XGBClassifier(random_state=random_state), xgb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
xgb_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)

# %%
print(xgb_corr_gt1_scaled_grid.best_params_)

# %%
xgb_corr_gt1_scaled = XGBClassifier(**xgb_corr_gt1_scaled_grid.best_params_, random_state=random_state)
xgb_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_test, xgb_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        xgb_corr_gt1_scaled,
        f"XGBClassifier {xgb_corr_gt1_scaled_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        xgb_corr_gt1_scaled,
        f"XGBClassifier {xgb_corr_gt1_scaled_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        xgb_corr_gt1_scaled,
        f"XGBClassifier {xgb_corr_gt1_scaled_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmark_results

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler_standard, pca=pca_standard)
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
xgb_pca_baseline = XGBClassifier(random_state=random_state)
xgb_pca_baseline.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_val, xgb_pca_baseline.predict(X_pca_val)))

# %%
xgb_pca_grid = GridSearchCV(XGBClassifier(random_state=random_state), xgb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
xgb_pca_grid.fit(X_pca_val, y_pca_val)

# %%
print(xgb_pca_grid.best_params_)

# %%
xgb_pca = XGBClassifier(**xgb_pca_grid.best_params_, random_state=random_state)
xgb_pca.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_test, xgb_pca.predict(X_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        xgb_pca,
        f"XGBClassifier {xgb_pca_grid.best_params_}",
        "Known attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        xgb_pca,
        f"XGBClassifier {xgb_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        xgb_pca,
        f"XGBClassifier {xgb_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1_standard)
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
xgb_corr_gt1_pca_baseline = XGBClassifier(random_state=random_state)
xgb_corr_gt1_pca_baseline.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_val, xgb_corr_gt1_pca_baseline.predict(X_corr_gt1_pca_val)))

# %%
xgb_corr_gt1_pca_grid = GridSearchCV(XGBClassifier(random_state=random_state), xgb_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
xgb_corr_gt1_pca_grid.fit(X_corr_gt1_pca_val, y_corr_gt1_pca_val)

# %%
print(xgb_corr_gt1_pca_grid.best_params_)

# %%
xgb_corr_gt1_pca = XGBClassifier(**xgb_corr_gt1_pca_grid.best_params_, random_state=random_state)
xgb_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_test, xgb_corr_gt1_pca.predict(X_corr_gt1_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        xgb_corr_gt1_pca,
        f"XGBClassifier {xgb_corr_gt1_pca_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        xgb_corr_gt1_pca,
        f"XGBClassifier {xgb_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        xgb_corr_gt1_pca,
        f"XGBClassifier {xgb_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmark_results

# %% [markdown]
# ## Random Forest

# %%
random_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# %% [markdown]
# ### All features scaled

# %%
df_scaled = pipeline_scaled(df=df, scaler=scaler_standard)
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

RF_scaled_baseline = RandomForestClassifier(random_state=random_state)
RF_scaled_baseline.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_val, RF_scaled_baseline.predict(X_scaled_val)))

# %%
RF_scaled_grid = GridSearchCV(RandomForestClassifier(random_state=random_state), random_forest_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
RF_scaled_grid.fit(X_scaled_val, y_scaled_val)

# %%
print(RF_scaled_grid.best_params_)

# %%
RF_scaled = RandomForestClassifier(**RF_scaled_grid.best_params_, random_state=random_state)
RF_scaled.fit(X_scaled_train, y_scaled_train)

# %%
print(classification_report(y_scaled_test, RF_scaled.predict(X_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        RF_scaled,
        f"Random Forest {RF_scaled_grid.best_params_}",
        "Known attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        RF_scaled,
        f"Random Forest {RF_scaled_grid.best_params_}",
        "Similar attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        RF_scaled,
        f"Random Forest {RF_scaled_grid.best_params_}",
        "New attacks",
        "All features scaled",
        pipeline_scaled,
        scaler=scaler_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 scaled

# %%
df_corr_gt1_scaled = pipeline_corr_gt1_scaled(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1)
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
RF_corr_gt1_scaled_baseline = RandomForestClassifier(random_state=random_state)
RF_corr_gt1_scaled_baseline.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_val, RF_corr_gt1_scaled_baseline.predict(X_corr_gt1_scaled_val)))

# %%
RF_corr_gt1_scaled_grid = GridSearchCV(RandomForestClassifier(random_state=random_state), random_forest_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
RF_corr_gt1_scaled_grid.fit(X_corr_gt1_scaled_val, y_corr_gt1_scaled_val)

# %%
print(RF_corr_gt1_scaled_grid.best_params_)

# %%
RF_corr_gt1_scaled = RandomForestClassifier(**RF_corr_gt1_scaled_grid.best_params_, random_state=random_state)
RF_corr_gt1_scaled.fit(X_corr_gt1_scaled_train, y_corr_gt1_scaled_train)

# %%
print(classification_report(y_corr_gt1_scaled_test, RF_corr_gt1_scaled.predict(X_corr_gt1_scaled_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        RF_corr_gt1_scaled,
        f"Random Forest {RF_corr_gt1_scaled_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        RF_corr_gt1_scaled,
        f"Random Forest {RF_corr_gt1_scaled_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        RF_corr_gt1_scaled,
        f"Random Forest {RF_corr_gt1_scaled_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features scaled",
        pipeline_corr_gt1_scaled,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1
        )

# %%
benchmark_results

# %% [markdown]
# ### All features with 95% PCA

# %%
df_pca = pipeline_pca(df=df, scaler=scaler_standard, pca=pca_standard)
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
RF_pca_baseline = RandomForestClassifier(random_state=random_state)
RF_pca_baseline.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_val, RF_pca_baseline.predict(X_pca_val)))

# %%
RF_pca_grid = GridSearchCV(RandomForestClassifier(random_state=random_state), random_forest_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
RF_pca_grid.fit(X_pca_val, y_pca_val)

# %%
print(RF_pca_grid.best_params_)

# %%
RF_pca = RandomForestClassifier(**RF_pca_grid.best_params_, random_state=random_state)
RF_pca.fit(X_pca_train, y_pca_train)

# %%
print(classification_report(y_pca_test, RF_pca.predict(X_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        RF_pca,
        f"Random Forest {RF_pca_grid.best_params_}",
        "Known attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        RF_pca,
        f"Random Forest {RF_pca_grid.best_params_}",
        "Similar attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        RF_pca,
        f"Random Forest {RF_pca_grid.best_params_}",
        "New attacks",
        "All features with 95% PCA",
        pipeline_pca,
        scaler=scaler_standard,
        pca=pca_standard
        )

# %%
benchmark_results

# %% [markdown]
# ### Features with |correlation| > 0.1 with 95% PCA

# %%
df_corr_gt1_pca = pipeline_corr_gt1_pca(df=df, scaler=scaler_standard_gt1, cols=cols_corr_gt1, pca=pca_corr_gt1_standard)
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
RF_corr_gt1_pca_baseline = RandomForestClassifier(random_state=random_state)
RF_corr_gt1_pca_baseline.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_val, RF_corr_gt1_pca_baseline.predict(X_corr_gt1_pca_val)))

# %%
RF_corr_gt1_pca_grid = GridSearchCV(RandomForestClassifier(random_state=random_state), random_forest_params, cv=cv, n_jobs=n_jobs, verbose=verbose)
RF_corr_gt1_pca_grid.fit(X_corr_gt1_pca_val, y_corr_gt1_pca_val)

# %%
print(RF_corr_gt1_pca_grid.best_params_)

# %%
RF_corr_gt1_pca = RandomForestClassifier(**RF_corr_gt1_pca_grid.best_params_, random_state=random_state)
RF_corr_gt1_pca.fit(X_corr_gt1_pca_train, y_corr_gt1_pca_train)

# %%
print(classification_report(y_corr_gt1_pca_test, RF_corr_gt1_pca.predict(X_corr_gt1_pca_test)))

# %%
benchmarkAndUpdateResult(
        df_known_attacks,
        RF_corr_gt1_pca,
        f"Random Forest {RF_corr_gt1_pca_grid.best_params_}",
        "Known attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_similar_attacks,
        RF_corr_gt1_pca,
        f"Random Forest {RF_corr_gt1_pca_grid.best_params_}",
        "Similar attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmarkAndUpdateResult(
        df_new_attacks,
        RF_corr_gt1_pca,
        f"Random Forest {RF_corr_gt1_pca_grid.best_params_}",
        "New attacks",
        "|correlation| > 0.1 features with 95% PCA",
        pipeline_corr_gt1_pca,
        scaler=scaler_standard_gt1,
        cols=cols_corr_gt1,
        pca=pca_corr_gt1_standard
        )

# %%
benchmark_results

# %%
benchmark_results.to_csv("benchmark_results.csv", index=False)
