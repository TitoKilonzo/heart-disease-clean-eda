# Heart Disease Prediction: Clean → Encode → Train → Evaluate

A Random Forest pipeline on the Heart Disease dataset, with an emphasis on not trusting a "clean-looking" file blindly and not letting class balancing leak into the test set.

## What's in the file

The CSV already has continuous columns scaled to 0–1 and categorical columns one-hot encoded, from earlier work on this dataset. The notebook doesn't assume that — every cleaning and encoding step checks the data first and only acts if it's actually needed.

## The core issue this notebook fixes

The file also contains duplicate rows from an earlier class-balancing step (random oversampling of the minority class, applied to the whole dataset before any split). That's a real risk: if you split after balancing, a duplicated patient record can land in both the training set and the test set, and your evaluation numbers stop meaning anything.

This notebook:
1. Removes the duplicates first, recovering the true 284 "No Disease" vs. 16 "Disease" split.
2. Splits into train/test on that honest, imbalanced data (stratified, so the rare class isn't accidentally absent from either side).
3. Balances only the training set afterward. The test set is never touched again until evaluation.

## Results from the included run

| Metric | Value |
|---|---|
| Duplicates removed | 268 (568 → 300 rows) |
| Train / test split | 240 / 60, stratified |
| Train accuracy (balanced) | 1.000 |
| Test accuracy (untouched) | 0.967 |
| Recall, "Disease" class | 0.33 (caught 1 of 3) |
| Precision, "Disease" class | 1.00 |

That gap between 96.7% accuracy and 33% recall on the class that actually matters clinically is the point of the exercise: accuracy alone would make this look like a strong model, while the confusion matrix shows it's missing most of the real Disease cases. With only 3 Disease patients in this test set, treat that recall figure as illustrative rather than a stable estimate — missing or catching one more patient swings it by ±33 points.

Top features by importance: Max Heart Rate, Cholesterol, Exercise-Induced Angina.

## Repo contents

```
Heart_Disease_Clean_Train_Evaluate.ipynb   the notebook, with outputs from a real run
Heart_Disease_Dataset.csv                  the dataset (scaled, encoded, pre-balancing)
requirements.txt
```

## Running it

### VS Code

1. Install the Python and Jupyter extensions if you don't have them.
2. Clone the repo and open the folder in VS Code.
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Open `Heart_Disease_Clean_Train_Evaluate.ipynb`, select the `.venv` interpreter as the kernel (top right), and Run All.

The notebook detects that it isn't running in Colab and automatically reads `Heart_Disease_Dataset.csv` from the same folder — no path editing needed.

### Google Colab

1. Upload `Heart_Disease_Dataset.csv` to `MyDrive/machine_learning/` in your Google Drive.
2. Upload the notebook to Colab (File → Upload notebook), or open it directly from GitHub (File → Open notebook → GitHub → paste the repo URL).
3. Runtime → Run all. It will prompt you to mount your Drive and then read the file from `MyDrive/machine_learning/Heart_Disease_Dataset.csv` automatically.

The notebook detects the environment itself (Colab vs. local), so the same file runs correctly either way.

## Tools

Python, pandas, scikit-learn (`RandomForestClassifier`, `train_test_split`, `confusion_matrix`, `classification_report`).
