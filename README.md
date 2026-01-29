# Student Performance Prediction Model

## Project Overview

This project uses machine learning methods to predict student final results (`final_result`), classified into four levels (0, 1, 2, 3). The project employs Genetic Algorithm (GA) for feature selection and compares the performance of Random Forest and XGBoost models.

## Dataset

| Dataset | Samples | Features | Classes | Training Set | Validation Set |
|---------|---------|----------|---------|--------------|----------------|
| B_group | 12,488 | 38 | 4 | 9,990 | 2,498 |
| J_group | 20,105 | 38 | 4 | 16,084 | 4,021 |

**Target Variable Distribution (Class Imbalance):**
- Class 0: ~20% (at-risk of failing)
- Class 1: ~31% (pass)
- Class 2: ~40% (good)
- Class 3: ~9.5% (excellent)

## Tech Stack

- **Python 3.x**
- **pandas** - Data processing
- **numpy** - Numerical computation
- **scikit-learn** - Model training and evaluation
- **XGBoost** - Gradient boosting model
- **matplotlib** - Visualization
- **imbalanced-learn** - Class imbalance handling (optional)

## Project Structure

```
Code&Data/
├── Prediction_Modelling.ipynb    # Main model training script
├── Data/processed/
│   ├── B_group.csv              # B group dataset
│   └── J_group.csv              # J group dataset
├── Data/raw/                    # Raw data files
└── feature_engineering.py       # GA feature selection module
```

## Key Features

### 1. Data Preparation (`prepare_train_data`)
- Automatically detects and removes non-numeric columns
- Supports stratified sampling (stratified split)
- Automatic integer conversion of target variable

### 2. Feature Selection (GAFeatureSelector)
- Uses Genetic Algorithm for feature selection
- Default parameters: population size 40, 25 generations
- Minimum selected features: 10

### 3. Model Training

Supports two models:

#### Random Forest
- Hyperparameter tuning: number of trees, max depth, min samples split, etc.
- Automatic class weight balancing (`class_weight='balanced'`)

#### XGBoost
- Supports multiclass and binary classification
- Adjustable learning rate, tree depth, regularization parameters
- Supports class imbalance handling (`scale_pos_weight`)

### 4. Confusion Matrix Visualization
- Blue-white color scheme
- Displays percentage and sample count
- Automatic text color adjustment based on cell intensity

## Model Performance Comparison

### B_group Dataset

| Model | Validation Accuracy | Best CV Score |
|-------|---------------------|---------------|
| Random Forest + GA | 74.14% | 73.12% |
| XGBoost + GA | 75.26% | 73.51% |

### J_group Dataset

| Model | Validation Accuracy | Best CV Score |
|-------|---------------------|---------------|
| Random Forest + GA | 77.69% | 77.24% |
| XGBoost + GA | 78.94% | 77.79% |

## Key Findings

1. **Class Imbalance Impact**: Class 3 (excellent) has the fewest samples, resulting in lower recognition rates
2. **Class 2 Dominance**: Due to having the most samples, the model tends to predict Class 2
3. **XGBoost Performs Better**: Outperforms Random Forest on both datasets
4. **J_group Shows Better Performance**: Larger dataset leads to better generalization

## Top Important Features

1. `weighted_score` - Weighted score
2. `total_weight_attempted` - Total weight attempted
3. `n_assessments` - Number of assessments
4. `total_clicks` - Total clicks
5. `studied_credits` - Credits studied

## How to Use

1. Ensure preprocessed data files are saved in `Data/processed/` directory
2. Run all cells in `Prediction_Modelling.ipynb`
3. Check output for model performance metrics and confusion matrix

## Potential Improvements

- [ ] Use SMOTE to handle class imbalance
- [ ] Try additional feature engineering methods
- [ ] Implement ensemble learning (Stacking/Boosting)
- [ ] Automated hyperparameter tuning (Optuna)
- [ ] Add cross-validation for more robust evaluation
- [ ] Implement model interpretation (SHAP/LIME)

## License

This project is for educational purposes as part of the FAIDM WM9QG-15 Group Assessment.
