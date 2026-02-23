"""
LendingClub Credit Risk Default Prediction Project
==================================================
This script performs a complete data science pipeline:
1. Download dataset from Kaggle
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Machine Learning Model for Credit Risk Default Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: DOWNLOAD DATASET FROM KAGGLE
# ============================================================================

def download_kaggle_dataset():
    """
    Download LendingClub dataset from Kaggle.
    Note: You need to set up Kaggle API credentials:
    1. Go to Kaggle Account -> API -> Create New Token
    2. Save kaggle.json to ~/.kaggle/kaggle.json (or C:/Users/<username>/.kaggle/kaggle.json on Windows)
    """
    import kaggle
    
    print("=" * 80)
    print("STEP 1: Downloading LendingClub Dataset from Kaggle")
    print("=" * 80)
    
    # LendingClub dataset on Kaggle
    dataset_name = "wordsforthewise/lending-club"
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(dataset_name, path='./data', unzip=True)
        print(f"✓ Dataset downloaded successfully!")
        
        # Find the CSV file
        data_dir = Path('./data')
        csv_files = list(data_dir.glob('*.csv'))
        
        if csv_files:
            print(f"✓ Found CSV file: {csv_files[0]}")
            return str(csv_files[0])
        else:
            print("⚠ No CSV file found. Please check the downloaded files.")
            return None
            
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("\nOr use a sample dataset path if you already have it.")
        return None


# ============================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================================

def load_and_clean_data(file_path=None):
    """
    Load and clean the LendingClub dataset.
    Handles: null values, duplicates, and data types.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Data Loading and Cleaning")
    print("=" * 80)
    
    # If no file path provided, try to find it
    if file_path is None:
        data_dir = Path('./data')
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            file_path = str(csv_files[0])
        else:
            print("⚠ Please provide the path to the LendingClub CSV file")
            return None
    
    # Load dataset
    print(f"\nLoading dataset from: {file_path}")
    try:
        # Load in chunks if file is large
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    # Display initial info
    print("\n--- Initial Dataset Info ---")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing values per column:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0].head(20))
    
    # Handle duplicates
    print(f"\n--- Duplicate Handling ---")
    initial_rows = len(df)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicates} duplicate rows")
        print(f"  Rows before: {initial_rows:,}, after: {len(df):,}")
    
    # Identify target variable (loan_status is typically the target)
    # Common values: 'Fully Paid', 'Charged Off', 'Default', etc.
    if 'loan_status' in df.columns:
        print(f"\n--- Target Variable (loan_status) Distribution ---")
        print(df['loan_status'].value_counts())
        
        # Create binary target: 1 for default/charged off, 0 for fully paid
        df['is_default'] = df['loan_status'].apply(
            lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)', 
                                'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'] 
            else 0 if x in ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'] 
            else np.nan
        )
        print(f"\n✓ Created binary target variable 'is_default'")
        print(f"  Default (1): {df['is_default'].sum():,}")
        print(f"  Fully Paid (0): {(df['is_default'] == 0).sum():,}")
    
    # Handle data types
    print(f"\n--- Data Type Conversion ---")
    
    # Convert date columns
    date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"✓ Converted {col} to datetime")
            except:
                pass
    
    # Convert percentage columns (remove % sign and convert to float)
    pct_columns = [col for col in df.columns if 'pct' in col.lower() or 'rate' in col.lower()]
    for col in pct_columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.rstrip('%').astype(float) / 100.0
                print(f"✓ Converted {col} to float")
            except:
                pass
    
    # Convert numeric columns that are stored as strings
    numeric_columns = df.select_dtypes(include=['object']).columns
    for col in numeric_columns[:20]:  # Check first 20 object columns
        try:
            # Try to convert to numeric
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.5:  # If >50% can be converted
                df[col] = converted
                print(f"✓ Converted {col} to numeric")
        except:
            pass
    
    # Handle null values
    print(f"\n--- Null Value Handling ---")
    
    # Drop columns with >50% missing values
    threshold = 0.5
    cols_to_drop = missing_df[missing_df['Missing %'] > threshold * 100].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        df = df.drop(columns=cols_to_drop)
    
    # For remaining columns, we'll handle nulls during feature engineering
    print(f"\n✓ Data cleaning completed!")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df):
    """
    Perform comprehensive Exploratory Data Analysis.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Exploratory Data Analysis (EDA)")
    print("=" * 80)
    
    if df is None or len(df) == 0:
        print("✗ No data available for EDA")
        return
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Target Variable Analysis
    if 'is_default' in df.columns:
        print("\n--- Target Variable Analysis ---")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        df['is_default'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
        axes[0].set_title('Distribution of Loan Status (Default vs Fully Paid)')
        axes[0].set_xlabel('Is Default (1=Default, 0=Fully Paid)')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['Fully Paid', 'Default'], rotation=0)
        
        # Percentage pie chart
        df['is_default'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
                                                          autopct='%1.1f%%', colors=['green', 'red'])
        axes[1].set_title('Loan Status Distribution (%)')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('plots/01_target_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/01_target_distribution.png")
        plt.close()
    
    # 2. Numerical Features Analysis
    print("\n--- Numerical Features Analysis ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_default' in numeric_cols:
        numeric_cols.remove('is_default')
    
    if len(numeric_cols) > 0:
        # Select top 12 numeric columns for analysis
        top_numeric = numeric_cols[:12]
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, col in enumerate(top_numeric):
            if idx < len(axes):
                df[col].hist(bins=50, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/02_numerical_distributions.png")
        plt.close()
    
    # 3. Correlation Analysis
    print("\n--- Correlation Analysis ---")
    if len(numeric_cols) > 1:
        # Select most important numeric columns for correlation
        important_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 
                         'fico_range_high', 'revol_util', 'total_acc', 'is_default']
        available_cols = [col for col in important_cols if col in df.columns]
        
        if len(available_cols) > 2:
            corr_matrix = df[available_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Matrix of Key Features')
            plt.tight_layout()
            plt.savefig('plots/03_correlation_matrix.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: plots/03_correlation_matrix.png")
            plt.close()
    
    # 4. Default Rate by Key Features
    print("\n--- Default Rate Analysis by Key Features ---")
    if 'is_default' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Default rate by loan amount (binned)
        if 'loan_amnt' in df.columns:
            df['loan_amnt_bin'] = pd.cut(df['loan_amnt'], bins=5)
            default_by_loan = df.groupby('loan_amnt_bin')['is_default'].mean()
            default_by_loan.plot(kind='bar', ax=axes[0, 0], color='coral')
            axes[0, 0].set_title('Default Rate by Loan Amount')
            axes[0, 0].set_xlabel('Loan Amount Bins')
            axes[0, 0].set_ylabel('Default Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Default rate by interest rate (binned)
        if 'int_rate' in df.columns:
            df['int_rate_bin'] = pd.cut(df['int_rate'], bins=5)
            default_by_int = df.groupby('int_rate_bin')['is_default'].mean()
            default_by_int.plot(kind='bar', ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Default Rate by Interest Rate')
            axes[0, 1].set_xlabel('Interest Rate Bins')
            axes[0, 1].set_ylabel('Default Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Default rate by FICO score
        if 'fico_range_low' in df.columns:
            df['fico_bin'] = pd.cut(df['fico_range_low'], bins=5)
            default_by_fico = df.groupby('fico_bin')['is_default'].mean()
            default_by_fico.plot(kind='bar', ax=axes[1, 0], color='coral')
            axes[1, 0].set_title('Default Rate by FICO Score')
            axes[1, 0].set_xlabel('FICO Score Bins')
            axes[1, 0].set_ylabel('Default Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Default rate by purpose
        if 'purpose' in df.columns:
            default_by_purpose = df.groupby('purpose')['is_default'].mean().sort_values(ascending=False)
            default_by_purpose.plot(kind='barh', ax=axes[1, 1], color='coral')
            axes[1, 1].set_title('Default Rate by Loan Purpose')
            axes[1, 1].set_xlabel('Default Rate')
        
        plt.tight_layout()
        plt.savefig('plots/04_default_rate_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/04_default_rate_analysis.png")
        plt.close()
    
    # 5. Summary Statistics
    print("\n--- Summary Statistics ---")
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols[:10]].describe()
        print(summary_stats)
    
    print("\n✓ EDA completed! All plots saved in 'plots' directory.")


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):
    """
    Perform feature engineering on the dataset.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Feature Engineering")
    print("=" * 80)
    
    if df is None or len(df) == 0:
        print("✗ No data available for feature engineering")
        return None
    
    df_processed = df.copy()
    
    # 1. Handle missing values for important features
    print("\n--- Handling Missing Values ---")
    
    # Numerical columns: impute with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_default' in numeric_cols:
        numeric_cols.remove('is_default')
    
    imputer_numeric = SimpleImputer(strategy='median')
    df_processed[numeric_cols] = imputer_numeric.fit_transform(df_processed[numeric_cols])
    print(f"✓ Imputed {len(numeric_cols)} numerical columns with median")
    
    # Categorical columns: impute with mode or 'Unknown'
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols[:20]:  # Process first 20 categorical columns
        if df_processed[col].isnull().sum() > 0:
            mode_value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
            df_processed[col].fillna(mode_value, inplace=True)
    
    print(f"✓ Imputed categorical columns with mode")
    
    # 2. Create new features
    print("\n--- Creating New Features ---")
    
    # FICO score average
    if 'fico_range_low' in df_processed.columns and 'fico_range_high' in df_processed.columns:
        df_processed['fico_avg'] = (df_processed['fico_range_low'] + df_processed['fico_range_high']) / 2
        print("✓ Created: fico_avg")
    
    # Debt-to-Income ratio categories
    if 'dti' in df_processed.columns:
        df_processed['dti_category'] = pd.cut(df_processed['dti'], 
                                               bins=[0, 10, 20, 30, 100], 
                                               labels=['Low', 'Medium', 'High', 'Very High'])
        print("✓ Created: dti_category")
    
    # Loan amount to income ratio
    if 'loan_amnt' in df_processed.columns and 'annual_inc' in df_processed.columns:
        df_processed['loan_to_income'] = df_processed['loan_amnt'] / (df_processed['annual_inc'] + 1)
        print("✓ Created: loan_to_income")
    
    # Credit utilization ratio (if available)
    if 'revol_bal' in df_processed.columns and 'annual_inc' in df_processed.columns:
        df_processed['credit_utilization'] = df_processed['revol_bal'] / (df_processed['annual_inc'] + 1)
        print("✓ Created: credit_utilization")
    
    # Years since earliest credit line
    if 'earliest_cr_line' in df_processed.columns:
        df_processed['earliest_cr_line'] = pd.to_datetime(df_processed['earliest_cr_line'], errors='coerce')
        if 'issue_d' in df_processed.columns:
            df_processed['issue_d'] = pd.to_datetime(df_processed['issue_d'], errors='coerce')
            df_processed['credit_history_years'] = (
                (df_processed['issue_d'] - df_processed['earliest_cr_line']).dt.days / 365.25
            )
            print("✓ Created: credit_history_years")
    
    # 3. Encode categorical variables
    print("\n--- Encoding Categorical Variables ---")
    
    # Label encode important categorical columns
    important_categorical = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'emp_length', 'verification_status']
    label_encoders = {}
    
    for col in important_categorical:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"✓ Encoded: {col}")
    
    # 4. Select features for modeling
    print("\n--- Feature Selection ---")
    
    # Select important features for modeling
    feature_columns = []
    
    # Numerical features
    important_numeric = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_avg', 
                         'revol_util', 'total_acc', 'loan_to_income', 'credit_utilization',
                         'credit_history_years']
    feature_columns.extend([col for col in important_numeric if col in df_processed.columns])
    
    # Encoded categorical features
    encoded_categorical = [col + '_encoded' for col in important_categorical]
    feature_columns.extend([col for col in encoded_categorical if col in df_processed.columns])
    
    # Add any other numeric columns that might be useful
    remaining_numeric = [col for col in numeric_cols if col not in feature_columns and 
                         col not in ['fico_range_low', 'fico_range_high']]
    feature_columns.extend(remaining_numeric[:10])  # Add top 10 remaining numeric columns
    
    print(f"✓ Selected {len(feature_columns)} features for modeling")
    print(f"  Features: {', '.join(feature_columns[:15])}...")
    
    # Prepare final dataset
    if 'is_default' in df_processed.columns:
        X = df_processed[feature_columns].copy()
        y = df_processed['is_default'].copy()
        
        # Remove rows where target is NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n✓ Feature engineering completed!")
        print(f"  Final feature matrix shape: {X.shape}")
        print(f"  Target distribution: Default={y.sum()}, Fully Paid={(y==0).sum()}")
        
        return X, y, feature_columns
    
    return None, None, feature_columns


# ============================================================================
# STEP 5: MACHINE LEARNING MODEL
# ============================================================================

def train_ml_model(X, y):
    """
    Train machine learning models for credit risk default prediction.
    """
    print("\n" + "=" * 80)
    print("STEP 5: Machine Learning Model Training")
    print("=" * 80)
    
    if X is None or y is None:
        print("✗ No data available for modeling")
        return None
    
    # Handle remaining missing values
    X = X.fillna(X.median())
    
    # Split data
    print("\n--- Data Splitting ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")
    
    # Handle class imbalance with SMOTE
    print("\n--- Handling Class Imbalance ---")
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"✓ After SMOTE - Training set: {X_train_resampled.shape[0]:,} samples")
        print(f"  Class distribution: Default={y_train_resampled.sum()}, Fully Paid={(y_train_resampled==0).sum()}")
    except Exception as e:
        print(f"⚠ SMOTE failed: {e}. Using original training set.")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Scale features
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler")
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n--- Model Training ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_resampled)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_score': auc_score
        }
        
        print(f"✓ {name} trained")
        print(f"  AUC Score: {auc_score:.4f}")
    
    # Best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\n--- Best Model: {best_model_name} ---")
    print(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")
    
    # Detailed evaluation of best model
    print("\n--- Detailed Classification Report ---")
    y_pred_best = results[best_model_name]['predictions']
    print(classification_report(y_test, y_pred_best, target_names=['Fully Paid', 'Default']))
    
    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    # Plot results
    print("\n--- Generating Model Evaluation Plots ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Fully Paid', 'Default'],
                yticklabels=['Fully Paid', 'Default'])
    axes[0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        axes[1].plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves Comparison')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/05_model_evaluation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/05_model_evaluation.png")
    plt.close()
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print("\n--- Top 10 Feature Importances ---")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('plots/06_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/06_feature_importance.png")
        plt.close()
    
    print("\n✓ Model training completed!")
    return best_model, scaler, results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that runs the complete pipeline.
    """
    print("\n" + "=" * 80)
    print("LENDINGCLUB CREDIT RISK DEFAULT PREDICTION PROJECT")
    print("=" * 80)
    
    # Step 1: Download dataset
    csv_path = download_kaggle_dataset()
    
    # If download failed, try to use existing file or ask user for path
    if csv_path is None:
        print("\nPlease provide the path to your LendingClub CSV file:")
        print("Example: data/accepted_2007_to_2018Q4.csv")
        csv_path = input("Enter path (or press Enter to skip): ").strip()
        if not csv_path:
            csv_path = None
    
    # Step 2: Load and clean data
    df = load_and_clean_data(csv_path)
    
    if df is None:
        print("\n✗ Cannot proceed without data. Please ensure dataset is available.")
        return
    
    # Step 3: EDA
    perform_eda(df)
    
    # Step 4: Feature Engineering
    X, y, feature_columns = feature_engineering(df)
    
    if X is None or y is None:
        print("\n✗ Cannot proceed without features and target. Please check data.")
        return
    
    # Step 5: Train ML Model
    model, scaler, results = train_ml_model(X, y)
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nAll outputs saved:")
    print("  - Plots: ./plots/")
    print("  - Model: Trained and ready for predictions")
    print("\nTo make predictions on new data:")
    print("  predictions = model.predict(scaler.transform(new_data))")


if __name__ == "__main__":
    main()
