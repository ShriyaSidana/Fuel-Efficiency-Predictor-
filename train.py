import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ---------- Load ----------
df = pd.read_csv("data/CO2_Emissions_Canada.csv").drop_duplicates()

num_cols = ["Engine Size(L)", "Cylinders"]
cat_cols = ["Vehicle Class", "Transmission", "Fuel Type"]
features = num_cols + cat_cols
target = "Fuel Consumption Comb (L/100 km)"

X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Preprocessing: one-hot encode text columns, pass numbers through ----------
def make_pipeline(model):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")
    return Pipeline([("prep", prep), ("model", model)])

# ---------- Three regression models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

# ---------- Train + compare ----------
rows, trained = [], {}
for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })
    trained[name] = pipe

results = pd.DataFrame(rows).sort_values("RMSE").round(3)
print(results)
results.to_csv("metrics.csv", index=False)

# ---------- Save the best model ----------
best_name = results.iloc[0]["Model"]
best = trained[best_name]
joblib.dump(best, "model.pkl")
print("Saved:", best_name)

# ---------- Feature importance ----------
imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=42)
importance = pd.DataFrame({"Feature": features, "Importance": imp.importances_mean})
importance.sort_values("Importance", ascending=False).to_csv("importance.csv", index=False)
print(importance.sort_values("Importance", ascending=False))