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
from benchmarkUtils import Benchmark
pd.set_option("display.width", 10000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
bu = Benchmark()
bu.import_df("./benchmark_results.csv")
import os
partial_tables = "./partial-tables"
os.makedirs(partial_tables, exist_ok=True)

# %%
common_latex_args = {
    "index":False,
    "escape":True,
    "formatters":{
        "Info": lambda s: s.replace("|correlation| > 0.1", "$|corr| > 0.1$")
    },
    "float_format":"{:.4f}".format
}


# %%
def pretty_print_model(model, to_latex=False):
    df = bu.display(model=model)
    df = df.rename(columns={"Time per data per iter": "Detection time (ms)"})
    df["Dataset"] = df["Dataset"].str.replace(" attacks", "")
    df = df.rename(columns={"Info":"Pipeline"})
    df = df.drop(columns=["Data size", "Model"])
    df = df.round(5)
    if not to_latex:
        df["Pipeline"] = df["Pipeline"].str.replace("correlation", "corr")
        df = df.set_index(["Pipeline", "Dataset"])
        display(df)
    else:
        df["Pipeline"] = df["Pipeline"].str.replace("|correlation| > 0.1", "$|corr| > 0.1$")
        df = df.set_index(["Pipeline", "Dataset"])
        latex_args = {
            "escape":True,
            "float_format":"{:.4f}".format,
        }
        latex=df.to_latex(**latex_args)
        latex = latex.replace("\$","$")
        print(latex)

def pretty_print_model_params(model, to_latex=False):
    df = bu.display(model=model)
    params=df[["Model", "Info"]].drop_duplicates().reset_index(drop=True).values
    param_df = pd.DataFrame(columns=["Pipeline", *eval("{"+params[0][0].split("{")[1]).keys()])
    for param in params:
        # if not to_latex:
        #     print("Pipeline: ", param[1])
        #     print("Params  : ", param[0])
        #     print()
        param_df.loc[len(param_df)] = [param[1], *eval("{"+param[0].split("{")[1]).values()]
    param_df = param_df.set_index("Pipeline")
    if not to_latex:
        display(param_df)
    else:
        latex_args = {
            "escape":True,
            "float_format":"{:.4f}".format,
        }
        latex=param_df.to_latex(**latex_args)
        latex = latex.replace("\$","$")
        print(latex)

# %% [markdown]
# ### Results of each model

# %%
pretty_print_model(model="SVM")
pretty_print_model_params(model="SVM")

# %%
pretty_print_model(model="GaussianNB")
pretty_print_model_params(model="GaussianNB")

# %%
pretty_print_model(model="Logistic Regression")
pretty_print_model_params(model="Logistic Regression")

# %%
pretty_print_model(model="XGBClassifier")
pretty_print_model_params(model="XGBClassifier")

# %%
pretty_print_model(model="Random Forest")
pretty_print_model_params(model="Random Forest")


# %%
def pretty_print_df(df, to_latex=False):
    tmp = df.copy()
    tmp["Model"] = tmp["Model"].str.replace("{.*}", "", regex=True)
    tmp["Info"] = tmp["Info"].str.replace("correlation", "corr")
    tmp = tmp.set_index("Model")
    tmp = tmp.drop(columns=["Dataset", "Data size"])
    tmp = tmp.rename(columns={"Time per data per iter":"Detection time (ms)","Info":"Pipeline"})
    tmp = tmp.round(5)
    if not to_latex:
        display(tmp)
    else: 
        latex_args = common_latex_args
        latex_args["index"] = True
        tmp["Pipeline"] = tmp["Pipeline"].str.replace("|corr| > 0.1", "$|corr| > 0.1$")
        print(tmp.to_latex(multirow=True, multicolumn=True, **latex_args))

# %% [markdown]
# ### Results on New dataset

# %%
pretty_print_df(bu.display(dataset="New", sort_by="Time per data per iter", ascending=True, top=10))

# %%
pretty_print_df(bu.display(dataset="New", sort_by="F1", ascending=False, top=10))

# %%
pretty_print_df(bu.display(dataset="New", sort_by="Recall", ascending=False, top=5))

# %%
pretty_print_df(bu.display(dataset="New", sort_by="Precision", ascending=False, top=5))

# %%
pretty_print_df(bu.display(dataset="New", sort_by="Accuracy", ascending=False, top=5))
