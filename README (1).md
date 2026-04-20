# 🏠 House Price Prediction — Machine Learning Project

**Students:** Nirmit Arora , Bhanu Sharma , Bikash Bhusal , Adarsh Singh , Shreyanshi Tripathi ( Group Project )
**University:** IILM University, Greater Noida
**Program:** Bachelor of Computer Applications (BCA) — 4th Semester
**Subject:** Minor Project — Machine Learning

---

## 📌 Project Overview

This project builds a **Machine Learning model to predict house prices** based on various features of a house such as size, number of bedrooms, location, age, and condition.

The dataset used is a **real-world Washington State (USA) housing dataset** containing 4,600 house sale records. Three different ML models were trained, evaluated, and compared to find the best-performing one.

---

## 📁 Repository Structure

```
house-price-prediction/
│
├── house_price_prediction_v2.py   ← Main Python source code
├── data.csv                       ← Dataset (Washington State housing data)
├── model_analysis_v2.png          ← Output charts and visualizations
├── requirements.txt               ← Python libraries required
└── README.md                      ← Project documentation (this file)
```

---

## 📊 Dataset Description

| Property        | Details                          |
|----------------|----------------------------------|
| Source          | Washington State Housing Data    |
| Total Records   | 4,600 houses                     |
| Features        | 18 columns                       |
| Target Variable | `price` (house sale price in USD)|
| Location        | Washington State, USA            |

### Key Features Used

| Feature          | Description                                 |
|-----------------|---------------------------------------------|
| `sqft_living`    | Living area in square feet                  |
| `bedrooms`       | Number of bedrooms                          |
| `bathrooms`      | Number of bathrooms                         |
| `sqft_lot`       | Total lot/land size                         |
| `floors`         | Number of floors                            |
| `waterfront`     | Whether the house has a water view (0/1)    |
| `condition`      | Overall condition rating (1–5)              |
| `yr_built`       | Year the house was built                    |
| `city`           | City name (e.g., Seattle, Bellevue)         |

---

## ⚙️ Methodology

The project follows the standard **Machine Learning pipeline**:

### 1. Data Cleaning
- Removed 49 rows where `price = 0` (invalid data)
- Removed extreme outliers using the **IQR (Interquartile Range) method**
- Final clean dataset: **4,237 records**

### 2. Feature Engineering
Six new features were created from existing columns to improve model accuracy:

| New Feature              | How It Was Created                         |
|--------------------------|---------------------------------------------|
| `house_age`              | `2024 - yr_built`                           |
| `was_renovated`          | 1 if renovated, 0 if not                   |
| `years_since_renovation` | Years since last renovation                 |
| `total_rooms`            | `bedrooms + bathrooms`                      |
| `living_lot_ratio`       | `sqft_living / sqft_lot`                    |
| `above_ratio`            | `sqft_above / sqft_living`                  |

### 3. Preprocessing Pipeline
- **StandardScaler** — rescaled all numerical features to the same range
- **OneHotEncoder** — converted city and statezip text columns to numerical format correctly (avoiding the ranking problem of LabelEncoder)

### 4. Models Trained

| Model                     | Description                                                |
|--------------------------|------------------------------------------------------------|
| Linear Regression         | Finds a mathematical formula linking features to price     |
| Random Forest (Tuned)     | Ensemble of 100+ decision trees with hyperparameter tuning |
| Gradient Boosting         | Sequential tree-building where each tree fixes previous errors |

### 5. Hyperparameter Tuning
`RandomizedSearchCV` was used to automatically find the best settings for the Random Forest model by testing 20 different parameter combinations with 3-fold cross-validation.

### 6. Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **MAE** (Mean Absolute Error) | Average prediction error in dollars |
| **RMSE** (Root Mean Squared Error) | Error with heavier penalty for large mistakes |
| **R² Score** | How much of price variation the model explains (1.0 = perfect) |
| **Cross-Validation R²** | Model accuracy tested across 5 different data splits |

---

## 📈 Results

| Model                   | MAE ($)   | RMSE ($)  | R² Score | CV R²  |
|------------------------|-----------|-----------|----------|--------|
| Linear Regression       | 66,432    | 94,387    | **0.796**| 0.783  |
| Random Forest (Tuned)   | 77,522    | 106,808   | 0.739    | 0.722  |
| Gradient Boosting       | 69,558    | 97,432    | 0.783    | 0.766  |

### 🏆 Best Model: Linear Regression
- **R² Score: 0.796** — the model explains **79.6% of price variation**
- **MAE: $66,432** — on average, predictions are within $66K of the actual price
- Achieved through proper outlier removal, feature scaling, and OneHotEncoding

> **Improvement over v1:** R² improved from 0.60 → 0.80 and MAE dropped from $157,372 → $66,432 — a 58% reduction in error.

---

## 🔍 Key Insights

1. **`sqft_living` is the most important feature** — house size explains 12.5% of price variation alone
2. **Location matters** — `statezip` (17%) and `city` (14%) together are the strongest predictors
3. **Bedrooms matter less than expected** — only 0.5% importance; size is far more important than room count
4. **House age and renovation status** contributed meaningfully after feature engineering

---

## 🖼️ Output Visualizations

The file `model_analysis_v2.png` contains 6 charts:

1. **Gradient Boosting — Actual vs Predicted** scatter plot
2. **Linear Regression — Actual vs Predicted** scatter plot
3. **Residuals Plot** — checks for prediction bias
4. **Model Comparison Bar Chart** — R² scores side by side
5. **Feature Importance** — top 10 most influential features
6. **Cross-Validation Boxplot** — consistency check across 5 folds

---

## 🚀 How to Run

### Step 1 — Install required libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Place the dataset
Make sure `data.csv` is in the **same folder** as the Python script.

### Step 3 — Run the project
```bash
python house_price_prediction_v2.py
```

### Step 4 — View output
- Results will print in the terminal
- `model_analysis_v2.png` will be saved in the same folder

---

## 🛠️ Technologies Used

| Tool / Library   | Purpose                              |
|-----------------|--------------------------------------|
| Python 3         | Programming language                 |
| pandas           | Data loading and manipulation        |
| numpy            | Numerical computations               |
| matplotlib       | Data visualization / charts          |
| seaborn          | Statistical visualizations           |
| scikit-learn     | Machine learning models & evaluation |

---

## 📚 Concepts Demonstrated

- Exploratory Data Analysis (EDA)
- Outlier detection and removal (IQR method)
- Feature Engineering
- Data Preprocessing Pipelines
- StandardScaler and OneHotEncoder
- Linear Regression
- Random Forest (Ensemble Learning)
- Gradient Boosting
- Hyperparameter Tuning (RandomizedSearchCV)
- K-Fold Cross Validation
- Model evaluation (MAE, RMSE, R²)
- Feature Importance Analysis

---

## 📝 Sample Prediction

Given a house with these details:
- 3 bedrooms, 2 bathrooms
- 1,800 sq ft living area
- Built in 1995, no renovation
- Located in Seattle, WA

| Model              | Predicted Price |
|-------------------|----------------|
| Linear Regression  | $613,481        |
| Random Forest      | $538,662        |
| Gradient Boosting  | $571,052        |
| **Average**        | **$574,398**    |
