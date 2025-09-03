import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Step 1: Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("salary_data.csv")
print("✅ Dataset loaded:", df.shape)

# Step 2: Split features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Step 3: Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Step 4: Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ================================
# 🔒 OLD MODEL: Linear Regression
# ================================
# print("🚀 Training Linear Regression model...")
# linreg_model = Pipeline(steps=[
#     ("prep", preprocessor),
#     ("regressor", LinearRegression())
# ])
# linreg_model.fit(X_train, y_train)
# y_pred_linreg = linreg_model.predict(X_test)
# mse_linreg = mean_squared_error(y_test, y_pred_linreg)
# r2_linreg = r2_score(y_test, y_pred_linreg)
# print("📊 Linear Regression MSE:", mse_linreg)
# print("📈 Linear Regression R²:", r2_linreg)

# ================================
# ✅ NEW MODEL: Random Forest
# ================================

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build Random Forest pipeline
rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 7: Train model
print("🚀 Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("✅ Model trained")

# Step 8: Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("📊 Random Forest MSE:", mse_rf)
print("📈 Random Forest R²:", r2_rf)

# Step 9: Save model
joblib.dump(rf_model, "salary_model.pkl")
print("💾 Model saved as salary_model.pkl")

# Step 10: Feature importance
print("\n🔍 Top Features:")
ohe = rf_model.named_steps["prep"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical_cols)
feature_names = list(numerical_cols) + list(ohe_features)

rf = rf_model.named_steps["regressor"]
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:5]

for i in top_idx:
    print(f"{feature_names[i]}: {importances[i]:.3f}")