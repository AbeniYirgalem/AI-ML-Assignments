# Assignment 12 — Retail Clustering Analysis

This folder contains Assignment 12 for the AI/ML course. The goal is to analyze an online retail dataset and apply clustering techniques to segment customers.

## Files

- `retail_customer_analysis.py`: Main Python script that loads `Online Retail.xlsx`, performs preprocessing, feature engineering, and applies clustering (e.g., KMeans) to identify customer segments.
- `Online Retail.xlsx`: Original retail transactions dataset.

## Requirements

- Python 3.8+ (or compatible)
- pandas
- scikit-learn
- matplotlib
- openpyxl (for reading Excel files)

Install dependencies:

```powershell
pip install pandas scikit-learn matplotlib openpyxl
```

## How to run

1. Open PowerShell and change to this folder:

```powershell
cd Assignment_12_Retail_Clustering_Analysis
```

2. Run the script:

```powershell
python retail_customer_analysis.py
```

The script will print dataset summary statistics, display clustering diagnostics (e.g., elbow plot), and output cluster labels or visualizations.

## Notes

- Ensure `Online Retail.xlsx` is present in the same folder as the script.
- Adjust clustering parameters in the script for different numbers of clusters.

## License

Educational use only.
