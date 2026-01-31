# Insurance Claim Prediction Model

## 📋 Project Overview

This project builds a **predictive model** to determine whether a building will have an insurance claim during a certain period. The model analyzes building characteristics to predict the probability of at least one claim occurring over the insured period.

**Target Variable:**
- `1` = Building has at least one claim over the insured period
- `0` = Building has no claims over the insured period

---

## 🎯 Objective

As a Lead Data Analyst, the goal is to:
1. Perform comprehensive data preprocessing and cleaning
2. Conduct detailed exploratory data analysis (EDA)
3. Engineer relevant features for modeling
4. Build and compare multiple machine learning models
5. Evaluate model performance using various metrics
6. Select and recommend the best model for deployment

---

## 📊 Dataset

The dataset contains **7,160 building records** with the following features:

| Feature | Description |
|---------|-------------|
| Customer Id | Identification number for the policy holder |
| YearOfObservation | Year of observation for the insured policy |
| Insured_Period | Duration of insurance policy (e.g., 1 = full year, 0.5 = 6 months) |
| Residential | Whether the building is residential (1) or not (0) |
| Building_Painted | Whether the building is painted |
| Building_Fenced | Whether the building is fenced |
| Garden | Whether the building has a garden |
| Settlement | Building location (Urban/Rural) |
| Building Dimension | Size of the insured building in m² |
| Building_Type | Type of building (1, 2, 3, or 4) |
| Date_of_Occupancy | Date building was first occupied |
| NumberOfWindows | Number of windows in the building |
| Geo_Code | Geographical code of the insured building |
| **Claim** | **Target variable** (0 = no claim, 1 = has claim) |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/insurance-claim-prediction.git
cd insurance-claim-prediction
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook insurance_claim_prediction.ipynb
```

---

## 📁 Project Structure

```
insurance-claim-prediction/
│
├── README.md                              # Project documentation
├── insurance_claim_prediction.ipynb       # Main analysis notebook
├── requirements.txt                       # Python dependencies
│
├── data/
│   ├── Train_data.csv                    # Training dataset
│   └── Variable_Description.csv          # Feature descriptions
│
└── models/                                # Saved models (optional)
    ├── best_model.pkl
    └── scaler.pkl
```

---

## 🔍 Analysis Workflow

### 1. **Data Cleaning & Preprocessing**
- Handled missing values in `NumberOfWindows`
- Encoded categorical variables (Building_Painted, Building_Fenced, Garden, Settlement)
- Created `Building_Age` feature from `Date_of_Occupancy`
- Removed non-predictive identifier (`Customer Id`)

### 2. **Exploratory Data Analysis (EDA)**
- Target variable distribution analysis
- Numerical features distribution and outlier detection
- Correlation analysis with target variable
- Claim rate analysis by:
  - Building type
  - Year of observation
  - Building dimensions
  - Settlement type

### 3. **Feature Engineering**
- **Total_Features**: Sum of painted, fenced, and garden amenities
- **Dimension_Category**: Categorized building dimensions
- **Age_Category**: Categorized building age
- **InsuredPeriod_Category**: Categorized insurance period

### 4. **Model Implementation**

Implemented and compared **7 different models**:

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Simple tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential boosting algorithm
5. **XGBoost** - Optimized gradient boosting
6. **Support Vector Machine (SVM)** - Kernel-based classifier
7. **Naive Bayes** - Probabilistic classifier

### 5. **Model Evaluation**

**Metrics Used:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curves
- Cross-Validation (5-fold)

**Feature Importance Analysis** for tree-based models to understand key predictors.

---

## 📈 Key Findings

### Model Performance
The models were evaluated on multiple metrics, with **ROC-AUC** being the primary metric due to the classification nature of the problem.

**Top 3 Models:** *(Results will vary based on your data)*
1. Model 1 - ROC-AUC: X.XXX
2. Model 2 - ROC-AUC: X.XXX
3. Model 3 - ROC-AUC: X.XXX

### Important Features
*(Based on feature importance analysis)*
- Building Age
- Building Dimension
- Building Type
- Insured Period
- Geographic Location (Geo_Code)

### Insights
- Buildings in urban areas show different claim patterns than rural areas
- Older buildings tend to have higher claim rates
- Certain building types are more prone to claims
- Insurance period length correlates with claim probability

---

## 🚀 Usage

### Running the Analysis

1. **Open the Jupyter Notebook:**
```bash
jupyter notebook insurance_claim_prediction.ipynb
```

2. **Execute cells sequentially** to:
   - Load and explore data
   - Perform cleaning and preprocessing
   - Conduct EDA
   - Train multiple models
   - Evaluate and compare results

### Making Predictions

```python
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare new data
new_data = pd.DataFrame({...})  # Your building data

# Preprocess
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[:, 1]

print(f"Claim Prediction: {prediction[0]}")
print(f"Claim Probability: {probability[0]:.2%}")
```

---

## 📦 Dependencies

Key libraries used:
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning models and preprocessing
- **xgboost** - Gradient boosting implementation

See `requirements.txt` for complete list with versions.

---

## 🎓 Methodology Highlights

✅ **Comprehensive data cleaning** with proper missing value handling  
✅ **Detailed EDA** with multiple visualizations and statistical insights  
✅ **Feature engineering** to create meaningful predictors  
✅ **Multiple model comparison** (7 different algorithms)  
✅ **Rigorous evaluation** with multiple metrics and cross-validation  
✅ **Professional documentation** with clear explanations  

---

## 🔮 Future Improvements

- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV or RandomizedSearchCV
- **Feature Selection**: Use recursive feature elimination or LASSO regularization
- **Ensemble Methods**: Combine top models using voting or stacking
- **Class Imbalance Handling**: Implement SMOTE or class weights if needed
- **Model Deployment**: Create API endpoint for real-time predictions
- **Model Monitoring**: Track model performance over time and detect data drift

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided by [Insurance Company/Organization Name]
- Project requirements and guidance from [Course/Institution Name]
- Inspiration from various Kaggle competitions and data science projects

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out via:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/insurance-claim-prediction/issues)
- **Email**: your.email@example.com

---

**⭐ If you found this project helpful, please consider giving it a star!**
