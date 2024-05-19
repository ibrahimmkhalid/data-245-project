## DATA 245 Project
#### Fast Network Threat Detection based on the TRAbID dataset:
An Analysis of Various Machine Learning Models

---

[Youtube demo](https://www.youtube.com/watch?v=8fjJgPkYYII)

Files:
- `./data/download_data.sh` and `./data/make_data.sh` - Download [dataset](https://secplab.ppgia.pucpr.br/?q=trabid) and convert to small csv
- `./data/*.csv` - The actual datasets used in training and evaluation
- `./exploring.ipynb` - Perform EDA on dataset
- `./model_*.ipynb` - Testing notebook for the given model (also save ROC curves to `./report/figures/`)
- `./benchmark.ipynb` - perform benchmark on all models and save results to `./benchmark_results.csv`
- `./benchmarkUtils.py` - common class for performing benchmarks in all notebooks
- `./results_analysis.ipynb` - Display results from `./benchmark_results.csv` in a neatly readable format
- `./results_analysis_to_latex.py` - Python script to generate tables in `./report/partial-tables/`
- `./requirements.txt` - Minimal set of requirements to set up environment
- `./report/` - Latex source code for report
- `./presentation/` - Mid and final project presentation
- `./report.pdf` - Report output as of `report` submodule commit hash 4fa5f16 


---

Team members:
- Shrey Agarwal
- Ashwini Hiremath
- Ibrahim Khalid
- Harsh Shinde
- Justing Wang
