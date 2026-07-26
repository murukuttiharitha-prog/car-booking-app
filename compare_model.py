
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ---- Load / regenerate training data ----
np.random.seed(42)
car_types = ["Hatchback", "Sedan", "SUV"]
base_rate = {"Hatchback": 1200, "Sedan": 1800, "SUV": 2400}
per_km_rate = {"Hatchback": 8, "Sedan": 10, "SUV": 13}

rows = []
for _ in range(1500):
    car_type = np.random.choice(car_types)
    days = np.random.randint(1, 15)
    distance_km = np.random.randint(20, 1200)
    is_weekend = np.random.choice([0, 1])
    base = base_rate[car_type] * days
    distance_cost = per_km_rate[car_type] * distance_km
    weekend_surcharge = 0.15 * base if is_weekend else 0
    noise = np.random.normal(0, base * 0.05)
    fare = max(base + distance_cost + weekend_surcharge + noise, 500)
    rows.append([car_type, days, distance_km, is_weekend, round(fare, 2)])

df = pd.DataFrame(rows, columns=["car_type", "days", "distance_km", "is_weekend", "fare"])
encoder = LabelEncoder()
df["car_type_encoded"] = encoder.fit_transform(df["car_type"])

X = df[["car_type_encoded", "days", "distance_km", "is_weekend"]]
y = df["fare"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Models to compare ----
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

print(f"{'Model':<20} {'MAE':>10} {'R2 Score':>10}")
print("-" * 42)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"model": model, "mae": mae, "r2": r2}
    print(f"{name:<20} {mae:>10.2f} {r2:>10.3f}")

# ---- Pick best model by R2 ----
best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (R2 = {results[best_name]['r2']:.3f})")

# ---- Feature importance (for tree-based models) ----
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance (why the model predicts what it does):")
    print(importances)
elif hasattr(best_model, "coef_"):
    coefs = pd.Series(best_model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print("\nFeature coefficients (Linear Regression):")
    print(coefs)
