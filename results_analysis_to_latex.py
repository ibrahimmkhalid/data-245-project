import pandas as pd
pd.set_option("display.width", 10000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
df = pd.read_csv("./benchmark_results.csv")
import os
partial_tables = "./report/partial-tables"
os.makedirs(partial_tables, exist_ok=True)
df["Dataset"] = df["Dataset"].str.replace(" attacks", "")
df = df.rename(columns={"Info":"Pipeline"})
df = df.rename(columns={"Time per data per iter":"Time (ms)"})
df = df.round(5)
df = df.drop(columns=["Data size"])
df["Pipeline"] = df["Pipeline"].str.replace("All features scaled", "scaled")
df["Pipeline"] = df["Pipeline"].str.replace("|correlation| > 0.1 features scaled", "corr, scaled")
df["Pipeline"] = df["Pipeline"].str.replace("All features with 95% PCA", "PCA")
df["Pipeline"] = df["Pipeline"].str.replace("|correlation| > 0.1 features with 95% PCA", "corr, PCA")
df["Params"] = df["Model"].str.extract(r"(\{.*\})")
df["Model"] = df["Model"].str.replace(r"\{.*\}", "", regex=True).str.strip()
df_params = df[["Model", "Params", "Pipeline"]].drop_duplicates().reset_index(drop=True)
df = df.drop(columns=["Params"])
for model in df["Model"].unique():
    tmp = df[df["Model"] == model].drop(columns=["Model"])
    tmp["Pipeline"] = tmp["Pipeline"].str.replace("|corr| > 0.1", "$|corr| > 0.1$")
    tmp = tmp.set_index(["Pipeline", "Dataset"])
    caption = f"{model} results"
    label = f"table-{model.lower().replace(' ', '-')}-results"
    latex_args = {
        "index":True,
        "escape":True,
        "float_format":"{:.3f}".format,
        "multirow":True,
        "multicolumn":True,
        "position":"!htb",
        "caption":caption,
        "label":label
    }
    latex = tmp.to_latex(**latex_args)
    latex = latex.replace("\\$", "$")
    latex = latex.replace("\\begin{tabular}", "\\centering\n\\resizebox{\\linewidth}{!}{\\begin{tabular}")
    latex = latex.replace("\\end{tabular}", "\\end{tabular}\n}")
    latex = latex.replace("\\multirow[t]", "\\multirow[c]")
    with open(f"{partial_tables}/{label}.tex", "w") as f:
        f.write(latex)

for model in df_params["Model"].unique():
    tmp = df_params[df_params["Model"] == model].drop(columns=["Model"])
    new_cols = eval("{"+tmp["Params"].iloc[0].split("{")[1])
    for col in new_cols.keys():
        tmp[col] = tmp["Params"].apply(lambda x: eval("{"+x.split("{")[1])[col])
    tmp = tmp.drop(columns=["Params"])
    tmp = tmp.set_index("Pipeline")
    if "var_smoothing" in tmp.columns:
        tmp["var_smoothing"] = tmp["var_smoothing"].round(9)
        # force into scientific notation
        tmp["var_smoothing"] = tmp["var_smoothing"].apply(lambda x: "{:.2e}".format(x))
        tmp["var_smoothing"] = tmp["var_smoothing"].astype(str)
    tmp = tmp.T

    caption = f"{model} best hyperparameters"
    label = f"table-{model.lower().replace(' ', '-')}-params"
    latex_args = {
        "index":True,
        "escape":True,
        "float_format":"{:.2f}".format,
        "position":"!htb",
        "caption":caption,
        "label":label
    }
    latex = tmp.to_latex(**latex_args)
    latex = latex.replace("\\$", "$")
    latex = latex.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")

    with open(f"{partial_tables}/{label}.tex", "w") as f:
        f.write(latex)

for col in df.columns[3:]:
    tmp_ = df.drop(columns=[*[c for c in df.columns[3:] if c != col]])
    for dataset in ["New", "Similar", "Known"]:
        tmp = tmp_[tmp_["Dataset"] == dataset].drop(columns=["Dataset"])
        if col =="Time (ms)":
            tmp = tmp.sort_values(by=col, ascending=True)
        else:
            tmp = tmp.sort_values(by=col, ascending=False)
        tmp = tmp.reset_index()
        tmp["index"] = tmp.index + 1
        tmp.rename(columns={"index":"Rank"}, inplace=True)

        caption = f"Top 5 models by {col} on {dataset} data"
        label = f"table-top-5-{col.lower().replace(' ', '-')}-{dataset.lower()}"
        latex_args = {
            "index":False,
            "escape":True,
            "float_format":"{:.3f}".format,
            "position":"!htb",
            "caption":caption,
            "label":label
        }
        latex = tmp.head().to_latex(**latex_args)
        latex = latex.replace("\\$", "$")
        latex = latex.replace("\\begin{tabular}", "\\centering\n\\begin{tabular}")
        with open(f"{partial_tables}/{label}.tex", "w") as f:
            f.write(latex)

for file in os.listdir(partial_tables):
    print(f"\\input{{partial-tables/{file.split('.')[0]}}}")
