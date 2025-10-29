Loan Default Prediction — AI Project

Predicting borrower default using tabular credit data. This repo compares Decision Tree (DT), Feedforward Neural Network (FNN), and Random Forest (RF) on a real-world–style dataset and selects the best model based on rigorous evaluation.

Highlights

End-to-end pipeline: data checks → encoding/scaling → class-imbalance handling → train/validate/test → comparison

Models: DT, FNN (Keras), RF; RF selected with ≥85% test accuracy

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, confusion matrices, ROC/PR curves

Reproducible Colab/Notebook with clear markdown and a 5-sample demo predictions table

Dataset

Source: Kaggle Loan_default.csv (tabular; ~255k rows; 18 features; target Default)

Place the file at data/Loan_default.csv (not committed)

Features include: Age, Income, LoanAmount, CreditScore, InterestRate, LoanTerm, DTIRatio and categorical: Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner

Quickstart

# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt
# minimal: pandas numpy scikit-learn matplotlib seaborn
# neural: tensorflow (or keras)
# optional: imbalanced-learn, gradio

# 3) Run notebook / script
# Open the notebook in Jupyter/Colab OR:
python src/train_compare.py --data data/Loan_default.csv --test_size 0.3 --seed 42

What’s Inside

.
├─ data/                    # (ignored) place Loan_default.csv here
├─ src/
│  ├─ preprocess.py         # encoding, scaling, split, class weights
│  ├─ models.py             # DT, RF, FNN builders
│  ├─ eval.py               # metrics, confusion matrix, ROC/PR
│  └─ train_compare.py      # end-to-end training & comparison
├─ notebooks/
│  └─ A2_Intro_to_AI.ipynb  # main analysis (with markdown + figures)
├─ figures/                 # exported plots (ROC/PR, confusion matrices, feature importance)
└─ README.md

Results (Test Set)

Random Forest: Accuracy ≥ 0.85 (selected), strong weighted F1; robust on tabular data

FNN: ~0.70–0.72 accuracy; useful neural benchmark with class-weighted training

Decision Tree: Interpretable baseline; moderate accuracy

Reproducibility

Stratified train/test split with --seed 42

Identical preprocessing across models; class weights for imbalance

Figures saved to figures/ for the report/video

License & Citation

Add your license (e.g., MIT) in LICENSE

If you use the data, cite the Kaggle source for Loan_default.csv
