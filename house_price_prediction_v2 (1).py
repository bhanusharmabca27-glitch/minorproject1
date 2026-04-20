

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Preprocessing ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Models ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- Evaluation ---
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Hyperparameter Tuning ---
from sklearn.model_selection import RandomizedSearchCV

print("=" * 65)
print("   HOUSE PRICE PREDICTION v2.0 — UPGRADED & PROFESSIONAL")
print("=" * 65)



# STEP 2: LOAD DATA

df = pd.read_csv('data.csv')

print(f"\n[STEP 1] Dataset loaded — {df.shape[0]} rows, {df.shape[1]} columns")



# STEP 3: REMOVE BAD DATA (OUTLIERS)

# WHY: Extreme values like $26M mansions or 0-bedroom houses
# confuse the model. We keep only realistic, common houses.

# We use the IQR method:
#   - Q1 = 25th percentile (lower quarter)
#   - Q3 = 75th percentile (upper quarter)
#   - IQR = Q3 - Q1 (the middle 50% range)
#   - Remove anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR

print("\n[STEP 2] Cleaning Data & Removing Outliers")

# Remove zero prices (clearly wrong data)
df = df[df['price'] > 0]

# Remove price outliers using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

# Remove houses with 0 bedrooms (not a real house)
df = df[df['bedrooms'] > 0]

# Remove sqft_living outliers
Q1s = df['sqft_living'].quantile(0.25)
Q3s = df['sqft_living'].quantile(0.75)
IQRs = Q3s - Q1s
df = df[(df['sqft_living'] >= Q1s - 1.5 * IQRs) & (df['sqft_living'] <= Q3s + 1.5 * IQRs)]

print(f"  Rows after cleaning: {len(df)}  (removed unrealistic houses)")



# STEP 4: FEATURE ENGINEERING



print("\n[STEP 3] Feature Engineering — Creating New Useful Columns")

# House age is more intuitive than year built
df['house_age'] = 2024 - df['yr_built']

# Was the house ever renovated? (1 = yes, 0 = no)
df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)

# If renovated, how many years ago?
df['years_since_renovation'] = df.apply(
    lambda x: 2024 - x['yr_renovated'] if x['yr_renovated'] > 0 else x['house_age'],
    axis=1
)

# Total rooms (bedrooms + bathrooms combined)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Ratio of living area to lot size (how much of land is used)
df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)

# Ratio of above-ground area to total living area
df['above_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)

print("  Created 6 new features:")
print("    house_age, was_renovated, years_since_renovation")
print("    total_rooms, living_lot_ratio, above_ratio")



# STEP 5: SELECT FEATURES & TARGET

print("\n[STEP 4] Selecting Features")

# Columns we won't use (not useful for prediction)
drop_cols = ['date', 'street', 'country', 'yr_built', 'yr_renovated', 'price']

# Target variable (what we want to predict)
y = df['price']

# Feature matrix (inputs to the model)
X = df.drop(columns=drop_cols)

# Identify which columns are numbers vs text
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"  Numerical features  ({len(numerical_cols)}): {numerical_cols}")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"  Total features: {len(X.columns)}")



# STEP 6: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[STEP 5] Train/Test Split:")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")



# STEP 7: BUILD PREPROCESSING PIPELINE

# A Pipeline chains multiple steps together so they run automatically.
#
# For NUMERICAL columns:
#   StandardScaler → rescales all numbers to same range (mean=0, std=1)
#   WHY: sqft_living can be 3000 but bedrooms is only 3.
#        Without scaling, the model over-weights large numbers.
#
# For CATEGORICAL columns (city, statezip):
#   OneHotEncoder → creates separate 0/1 columns for each category
#   WHY: LabelEncoder gave city numbers (Seattle=32, Kent=17) which
#        implies a ranking that doesn't exist. OneHotEncoder avoids this.
#   Example: city column with [Seattle, Kent, Bellevue] becomes:
#            city_Seattle  city_Kent  city_Bellevue
#                1             0           0       ← Seattle
#                0             1           0       ← Kent
#                0             0           1       ← Bellevue

print("\n[STEP 6] Building Preprocessing Pipeline")

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

print("  StandardScaler  → applied to all numerical columns")
print("  OneHotEncoder   → applied to city and statezip columns")



# STEP 8: MODEL 1 — LINEAR REGRESSION (with Pipeline)

print("\n" + "=" * 65)
print("[STEP 7] Model 1: Linear Regression (with proper scaling)")
print("=" * 65)

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

# Cross-validation: test on 5 different splits, average the score
# This gives a more reliable accuracy than a single test split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_pipeline, X, y, cv=kf, scoring='r2')

print(f"\n  MAE            : ${mae_lr:,.0f}")
print(f"  RMSE           : ${rmse_lr:,.0f}")
print(f"  R² Score       : {r2_lr:.4f} ({r2_lr*100:.1f}%)")
print(f"  Cross-Val R²   : {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")



# STEP 9: MODEL 2 — RANDOM FOREST (Hyperparameter Tuned)

print("[STEP 8] Model 2: Random Forest (with Hyperparameter Tuning)")
print("=" * 65)

# Hyperparameter tuning = finding the BEST settings for the model
#
# Instead of guessing settings like n_estimators=100,
# we try many combinations and pick the best one automatically.
#
# n_estimators  = how many trees to build
# max_depth     = how deep each tree can grow
# min_samples_split = minimum samples needed to split a node
# max_features  = how many features each tree considers

print("\n  Searching for best hyperparameters (this takes ~30 seconds)...")

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators'    : [100, 200, 300],
    'model__max_depth'       : [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__max_features'    : ['sqrt', 'log2', 0.5]
}

# RandomizedSearchCV tries 20 random combinations (faster than trying all)
rf_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1      # use all CPU cores for speed
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print(f"\n  Best parameters found:")
for k, v in rf_search.best_params_.items():
    print(f"    {k.replace('model__', ''):<25}: {v}")

y_pred_rf = best_rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

cv_scores_rf = cross_val_score(best_rf, X, y, cv=kf, scoring='r2')

print(f"\n  MAE            : ${mae_rf:,.0f}")
print(f"  RMSE           : ${rmse_rf:,.0f}")
print(f"  R² Score       : {r2_rf:.4f} ({r2_rf*100:.1f}%)")
print(f"  Cross-Val R²   : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")



# STEP 10: MODEL 3 — GRADIENT BOOSTING

# Gradient Boosting is another powerful ensemble method.
# Unlike Random Forest (trees built independently),
# Gradient Boosting builds trees SEQUENTIALLY —
# each new tree tries to fix the mistakes of the previous one.
# This often gives the best accuracy of all three models.

print("\n" + "=" * 65)
print("[STEP 9] Model 3: Gradient Boosting Regressor")
print("=" * 65)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ))
])

gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_test)

mae_gb  = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb   = r2_score(y_test, y_pred_gb)

cv_scores_gb = cross_val_score(gb_pipeline, X, y, cv=kf, scoring='r2')

print(f"\n  MAE            : ${mae_gb:,.0f}")
print(f"  RMSE           : ${rmse_gb:,.0f}")
print(f"  R² Score       : {r2_gb:.4f} ({r2_gb*100:.1f}%)")
print(f"  Cross-Val R²   : {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")



# STEP 11: MODEL COMPARISON

print("\n" + "=" * 65)
print("[STEP 10] Final Model Comparison")
print("=" * 65)

results = pd.DataFrame({
    'Model'   : ['Linear Regression', 'Random Forest (Tuned)', 'Gradient Boosting'],
    'MAE ($)' : [mae_lr, mae_rf, mae_gb],
    'RMSE ($)': [rmse_lr, rmse_rf, rmse_gb],
    'R² Score': [r2_lr, r2_rf, r2_gb],
    'CV R²'   : [cv_scores_lr.mean(), cv_scores_rf.mean(), cv_scores_gb.mean()]
})

print(f"\n{results.to_string(index=False)}")

best_idx = results['R² Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
print(f"\n  🏆 Best Model: {best_model_name}")



# STEP 12: FEATURE IMPORTANCE

print("[STEP 11] Feature Importance (Random Forest)")
print("=" * 65)

# Get feature names after OneHotEncoding
rf_model_step = best_rf.named_steps['model']
prep_step = best_rf.named_steps['preprocessor']

ohe_features = prep_step.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = numerical_cols + list(ohe_features)

importances = pd.Series(rf_model_step.feature_importances_, index=all_features)

# Group OneHotEncoded columns back by original column name
grouped = {}
for feat, imp in importances.items():
    base = feat.split('_')[0] if '_' in feat else feat
    # Check if it's an OHE feature
    is_ohe = any(feat.startswith(c) for c in categorical_cols)
    key = feat.split('_')[0] if is_ohe else feat
    grouped[key] = grouped.get(key, 0) + imp

grouped_series = pd.Series(grouped).sort_values(ascending=False)

print("\nTop Features by Importance:")
for feat, imp in grouped_series.head(10).items():
    bar = "█" * int(imp * 60)
    print(f"  {feat:<25} {imp:.4f}  {bar}")



# STEP 13: GENERATE PROFESSIONAL CHARTS

fig = plt.figure(figsize=(16, 12))
fig.suptitle("House Price Prediction v2.0 — Model Analysis Dashboard",
             fontsize=15, fontweight='bold', y=0.98)

# Color palette
colors = {'lr': '#4C72B0', 'rf': '#55A868', 'gb': '#C44E52'}

# Chart 1: Actual vs Predicted — Best Model
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_gb, alpha=0.4, color=colors['gb'], s=15, label='Predictions')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect line')
ax1.set_xlabel("Actual Price ($)")
ax1.set_ylabel("Predicted Price ($)")
ax1.set_title(f"Gradient Boosting\nActual vs Predicted (R²={r2_gb:.3f})")
ax1.legend(fontsize=8)

# Chart 2: Actual vs Predicted — Linear Regression
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(y_test, y_pred_lr, alpha=0.4, color=colors['lr'], s=15)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel("Actual Price ($)")
ax2.set_ylabel("Predicted Price ($)")
ax2.set_title(f"Linear Regression\nActual vs Predicted (R²={r2_lr:.3f})")

# Chart 3: Residuals Plot (Gradient Boosting)
ax3 = fig.add_subplot(2, 3, 3)
residuals = y_test - y_pred_gb
ax3.scatter(y_pred_gb, residuals, alpha=0.4, color=colors['gb'], s=15)
ax3.axhline(y=0, color='red', linestyle='--', lw=2)
ax3.set_xlabel("Predicted Price ($)")
ax3.set_ylabel("Residual (Actual - Predicted)")
ax3.set_title("Residuals Plot\n(Good model: dots around zero line)")

# Chart 4: Model Comparison — R² Scores
ax4 = fig.add_subplot(2, 3, 4)
model_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
r2_vals = [r2_lr, r2_rf, r2_gb]
bar_colors = [colors['lr'], colors['rf'], colors['gb']]
bars = ax4.bar(model_names, r2_vals, color=bar_colors, width=0.5, edgecolor='white')
ax4.set_ylim(0, 1)
ax4.set_ylabel("R² Score")
ax4.set_title("Model Comparison — R² Score\n(Higher is Better)")
for bar, val in zip(bars, r2_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# Chart 5: Feature Importance
ax5 = fig.add_subplot(2, 3, 5)
top10 = grouped_series.head(10)
ax5.barh(top10.index[::-1], top10.values[::-1], color='#2ecc71', edgecolor='white')
ax5.set_xlabel("Importance Score")
ax5.set_title("Top 10 Feature Importances\n(Random Forest)")

# Chart 6: Cross-Validation Scores
ax6 = fig.add_subplot(2, 3, 6)
cv_data = [cv_scores_lr, cv_scores_rf, cv_scores_gb]
bp = ax6.boxplot(cv_data, patch_artist=True, labels=['Linear\nReg', 'Random\nForest', 'Gradient\nBoost'])
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel("R² Score")
ax6.set_title("Cross-Validation Scores\n(5-Fold, consistency check)")

plt.tight_layout()
plt.savefig('model_analysis_v2.png', dpi=150, bbox_inches='tight')
print("\n[Charts saved: model_analysis_v2.png]")



# STEP 14: PREDICT A SAMPLE HOUSE

print("\n" + "=" * 65)
print("[STEP 12] Predict Price for a Sample House")
print("=" * 65)

sample = pd.DataFrame([{
    'bedrooms'              : 3,
    'bathrooms'             : 2.0,
    'sqft_living'           : 1800,
    'sqft_lot'              : 5000,
    'floors'                : 1.0,
    'waterfront'            : 0,
    'view'                  : 0,
    'condition'             : 3,
    'sqft_above'            : 1800,
    'sqft_basement'         : 0,
    'city'                  : 'Seattle',
    'statezip'              : 'WA 98103',
    'house_age'             : 29,
    'was_renovated'         : 0,
    'years_since_renovation': 29,
    'total_rooms'           : 5.0,
    'living_lot_ratio'      : 1800 / 5001,
    'above_ratio'           : 1800 / 1801,
}])

pred_lr = lr_pipeline.predict(sample)[0]
pred_rf = best_rf.predict(sample)[0]
pred_gb = gb_pipeline.predict(sample)[0]

print(f"\n  House: 3 bed | 2 bath | 1800 sqft | Built 1995 | Seattle")
print(f"\n  Linear Regression  : ${pred_lr:,.0f}")
print(f"  Random Forest      : ${pred_rf:,.0f}")
print(f"  Gradient Boosting  : ${pred_gb:,.0f}")
print(f"\n  Average Prediction : ${np.mean([pred_lr, pred_rf, pred_gb]):,.0f}")

print("\n" + "=" * 65)
print("  PROJECT v2.0 COMPLETE!")
print("=" * 65)
