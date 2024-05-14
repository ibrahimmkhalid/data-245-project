import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Benchmark:
    def __init__(self, iter_n=10):
        self.__ITER__ = iter_n
        self.benchmark_results = self.make_df()

    def make_df(self):
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
        return benchmark_results

    def import_df(self, path):
        self.benchmark_results = pd.read_csv(path)

    def display(
        self,
        model=None,
        dataset=None,
        sort_by=None,
        ascending=False,
        top=None,
        show_model_params=True,
        reset_index=True,
    ):
        tmp = self.benchmark_results.copy()
        if model:
            tmp = tmp[tmp["Model"].str.contains(model)]
        if dataset:
            tmp = tmp[tmp["Dataset"].str.contains(dataset)]
        if sort_by:
            tmp = tmp.sort_values(by=sort_by, ascending=ascending)
        if not show_model_params:
            tmp["Model"] = tmp["Model"].str.replace(r" {.*}", "", regex=True)
        if top:
            tmp = tmp.head(top)
        if reset_index:
            tmp = tmp.reset_index(drop=True)
        return tmp

    def to_csv(self, path):
        self.benchmark_results.to_csv(path, index=False)

    def benchmarkAndUpdateResult(
        self, df, model, model_name, dataset_name, info, pipeline_fn, **pipeline_kwargs
    ):
        df_ = pipeline_fn(df=df, **pipeline_kwargs)
        X = df_[df_.columns[:-1]]
        y = df_[df_.columns[-1]]
        data_size = np.shape(X)[0]
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        start = time.perf_counter_ns()
        for _ in range(self.__ITER__):  # benchmark
            df_ = pipeline_fn(df=df, **pipeline_kwargs)
            X = df_[df_.columns[:-1]]
            model.predict(X)
        end = time.perf_counter_ns()
        time_per_data_per_iter = (end - start) / self.__ITER__ / 1000000
        self.benchmark_results.loc[len(self.benchmark_results)] = [
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
