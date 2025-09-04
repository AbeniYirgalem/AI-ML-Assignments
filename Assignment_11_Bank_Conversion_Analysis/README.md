# Assignment 11 — Bank Conversion Analysis

This folder contains Assignment 11 for the AI/ML course. The goal is to analyze a bank marketing dataset and build a model to predict customer conversion or campaign response.

## Files

- `bank_marketing_tree.py`: Main Python script that loads `bank-full.csv`, performs preprocessing, trains a decision tree classifier, and evaluates performance.
- `bank-full.csv`: Original bank marketing dataset (CSV).

## Requirements

- Python 3.8+ (or compatible)
- pandas
- scikit-learn
- matplotlib (optional, for plots)

Install dependencies:

```powershell
pip install pandas scikit-learn matplotlib
```

## How to run

1. Open PowerShell and change to this folder:

```powershell
cd Assignment_11_Bank_Conversion_Analysis
```

2. Run the script:

```powershell
python bank_marketing_tree.py
```

The script will print dataset summary statistics, training/validation scores, and may save or display evaluation plots.

## Notes

- Ensure `bank-full.csv` is present in the same folder as the script.
- If the script expects a different filename or path, update the script or move the CSV accordingly.

## License

Educational use only.
