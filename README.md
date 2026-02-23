# LendingClub Credit Risk Default Prediction

A comprehensive data science project for predicting credit risk defaults using LendingClub loan data.

## Project Overview

This project implements a complete machine learning pipeline for credit risk assessment:

1. **Data Acquisition**: Download LendingClub dataset from Kaggle
2. **Data Cleaning**: Handle null values, duplicates, and data type conversions
3. **Exploratory Data Analysis (EDA)**: Visualize and understand the data
4. **Feature Engineering**: Create new features and prepare data for modeling
5. **Machine Learning**: Train models to predict loan defaults

## Installation

### Prerequisites

- Python 3.8 or higher
- Kaggle API credentials (for automatic dataset download)

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Kaggle API (for automatic download):**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to "API" section and click "Create New Token"
   - This downloads `kaggle.json` file
   - Place it in:
     - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
     - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

3. **Alternative - Manual Download:**
   - If you prefer manual download, get the dataset from:
     [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   - Place the CSV file in the `./data/` directory

## Usage

### Option 1: Run Python Script

```bash
python lendingclub_credit_risk_analysis.py
```

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook lendingclub_credit_risk_analysis.ipynb
```

## Project Structure

```
.
├── lendingclub_credit_risk_analysis.py    # Main Python script
├── lendingclub_credit_risk_analysis.ipynb  # Jupyter notebook version
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── data/                                   # Dataset directory (created automatically)
└── plots/                                  # Generated visualizations (created automatically)
```

## Features

### 1. Data Download
- Automatic download from Kaggle using API
- Supports manual file path input
- Handles large datasets efficiently

### 2. Data Cleaning
- Removes duplicate records
- Handles missing values (drops columns with >50% missing, imputes others)
- Converts data types (dates, percentages, numeric strings)
- Creates binary target variable (default vs fully paid)

### 3. Exploratory Data Analysis
Generates comprehensive visualizations:
- Target variable distribution
- Numerical feature distributions
- Correlation matrix
- Default rate analysis by key features
- Summary statistics

### 4. Feature Engineering
- Creates new features:
  - FICO score average
  - Debt-to-income categories
  - Loan-to-income ratio
  - Credit utilization ratio
  - Credit history years
- Encodes categorical variables
- Handles missing values with appropriate imputation strategies

### 5. Machine Learning Models
Trains and compares multiple models:
- **Logistic Regression**: Baseline model
- **Random Forest**: Tree-based ensemble
- **Gradient Boosting**: Advanced ensemble method

**Model Evaluation:**
- AUC-ROC score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC curves comparison
- Feature importance (for tree-based models)

**Class Imbalance Handling:**
- Uses SMOTE (Synthetic Minority Oversampling Technique) to balance classes

## Output

The script generates:

1. **Plots Directory** (`./plots/`):
   - `01_target_distribution.png` - Loan status distribution
   - `02_numerical_distributions.png` - Feature distributions
   - `03_correlation_matrix.png` - Feature correlations
   - `04_default_rate_analysis.png` - Default rates by key features
   - `05_model_evaluation.png` - Confusion matrix and ROC curves
   - `06_feature_importance.png` - Top important features

2. **Console Output**:
   - Progress updates for each step
   - Data statistics and summaries
   - Model performance metrics

## Model Performance

The models are evaluated using:
- **AUC-ROC Score**: Measures the model's ability to distinguish between classes
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive samples
- **F1-Score**: Harmonic mean of precision and recall

## Customization

You can customize the script by:
- Adjusting feature selection in `feature_engineering()` function
- Modifying model parameters in `train_ml_model()` function
- Changing visualization styles and plots
- Adding additional models or evaluation metrics

## Troubleshooting

### Kaggle API Issues
- Ensure `kaggle.json` is in the correct location
- Verify API credentials are valid
- Check internet connection

### Memory Issues
- For large datasets, consider processing in chunks
- Reduce the number of features selected
- Use a subset of data for initial testing

### Missing Dependencies
- Run `pip install -r requirements.txt` again
- Ensure you're using Python 3.8+

## License

This project is for educational purposes. The LendingClub dataset is publicly available on Kaggle.

## References

- [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
