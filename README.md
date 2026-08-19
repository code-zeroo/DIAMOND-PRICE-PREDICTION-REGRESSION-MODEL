# Diamond Price Prediction — Regression Project

A regression project that predicts diamond prices from their physical and quality attributes, using the classic [Diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) (53,940 rows).

**Multiple regression models were trained and compared to identify the best-fit model for this dataset**, rather than relying on a single algorithm.

## Models Compared

| Model | R² Score | MSE | MAE |
|---|---|---|---|
| Linear Regression | 0.8827 | 1,828,839.11 | — |
| K-Nearest Neighbors Regressor | 0.9738 | 409,112.71 | 301.48 |
| Decision Tree Regressor | 0.9741 | 403,287.19 | — |
| **Random Forest Regressor** | **0.9810** | **296,514.82** | **271.47** |

**Best model: Random Forest Regressor**, selected based on the highest R² score and lowest error metrics on the test set after hyperparameter tuning.

## Workflow

1. **Data Cleaning** — checked for missing values, dropped the redundant index column.
2. **Statistical Feature Selection**
   - One-way ANOVA to test `cut`, `color`, and `clarity` against `price`
   - Pearson correlation for numerical features (`carat`, `x`, `y`, `z`) against `price`
   - Chi-square tests to check relationships between categorical features (`cut`, `color`, `clarity`)
3. **Preprocessing** — label encoding for categorical columns, feature scaling with `StandardScaler`, dropped low-signal columns (`table`, `depth`).
4. **Model Training & Comparison** — trained Linear Regression, Decision Tree, Random Forest, and KNN regressors.
5. **Hyperparameter Tuning** — used `GridSearchCV` (cross-validation) on each model to find its best configuration.
6. **Evaluation** — compared all tuned models on R², MSE, and MAE to select the best-performing one.

## Dataset

Features used:
- `carat`, `cut`, `color`, `clarity`, `x` (length), `y` (width), `z` (depth)
- Target: `price`

The dataset itself is not included in this repo — download it from Kaggle and update the file path in the notebook, or place it locally as `diamonds.csv` and update the read path accordingly.

## Repository Structure

```
.
├── Reg_project.ipynb   # Main notebook: EDA, hypothesis tests, model training & comparison
├── requirements.txt    # Python dependencies
├── README.md
└── .gitignore
```

## Getting Started

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Reg_project.ipynb
```

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- scipy (statistical tests)
- matplotlib, seaborn (visualization)

## License

This project is open source and available under the [MIT License](LICENSE).
