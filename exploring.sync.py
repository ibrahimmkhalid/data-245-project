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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
csv_path = './probe_known_attacks_ssmall.csv'

# %%
df = pd.read_csv(csv_path, low_memory=False)

# %%
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
y = df["class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(df.columns)-1)
X_pca = pca.fit_transform(X_scaled)

pca_cumsum = pca.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs Number of components")
plt.grid()
plt.xticks(range(0, len(df.columns)-1, 2))
plt.show()

# %% [markdown]
# - at n=21, we have 85% of the variance explained
# - at n=24, we have 90% of the variance explained
# - at n=27, we have 95% of the variance explained

# %%
X = df[cols_corr_gt1]
y = df["class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(cols_corr_gt1))
X_pca = pca.fit_transform(X_scaled)

pca_cumsum = pca.explained_variance_ratio_.cumsum()
plt.plot(pca_cumsum)
plt.xlabel("Number of components with |correlation| > 0.1")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs Number of components with |correlation| > 0.1")
plt.grid()
plt.xticks(range(0, len(cols_corr_gt1), 2))
plt.show()

# %% [markdown]
# - at n=10 of the selected features, we have 95% of the variance explained
