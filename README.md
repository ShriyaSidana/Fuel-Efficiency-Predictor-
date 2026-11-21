# ⛽ Fuel Efficiency Predictor | AutoFuel AI

**An interactive Machine Learning-based application for vehicle fuel consumption prediction, fuel cost estimation, vehicle comparison, and environmental impact analysis.**

AutoFuel AI leverages regression models to predict vehicle fuel efficiency using vehicle specifications. It provides insights into estimated fuel consumption, trip expenses, vehicle performance, and associated CO₂ emissions through an interactive Streamlit dashboard.

---

## 📌 Overview

Fuel efficiency is an important factor when evaluating vehicle performance, ownership costs, and environmental impact.

The **Fuel Efficiency Predictor** is a Machine Learning-powered application that estimates a vehicle's combined fuel consumption based on its technical specifications. It also helps users understand the financial and environmental implications of their vehicle choices.

The application combines predictive analytics, interactive visualizations, model evaluation, and explainable AI to make fuel efficiency analysis more accessible.

## ✨ Key Features

### 1. Fuel Consumption Prediction
- Predicts combined fuel consumption (L/100 km) using vehicle specifications.
- Provides fuel efficiency estimates in different units.
- Displays an A–G efficiency rating based on the benchmark dataset.
- Includes prediction uncertainty estimates and vehicle benchmark comparisons.

### 2. Trip Cost & Emissions Calculator
- Estimates the fuel required for a given trip distance.
- Calculates estimated trip expenses using configurable fuel prices.
- Supports one-way and round-trip calculations.
- Estimates trip-related CO₂ emissions.
- Visualizes fuel expenditure against travel distance.

### 3. Vehicle Comparison
- Compare two vehicles based on their technical specifications.
- Evaluate predicted fuel consumption and mileage.
- Compare annual fuel expenses and carbon emissions.
- Estimate cumulative fuel costs over different ownership periods.
- Identify the more economical vehicle based on estimated fuel expenses.

### 4. Model Insights & Explainable AI
- Compare multiple regression algorithms.
- Visualize MAE, RMSE, and R² evaluation metrics.
- Analyze feature importance using permutation importance.
- Understand individual predictions through feature contribution analysis.
- Visualize actual vs. predicted values and residual distributions.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning and model evaluation |
| Joblib | Model serialization and loading |
| Plotly | Interactive data visualizations |
| Jupyter Notebook | Exploratory data analysis and experimentation |

---

## 🤖 Machine Learning Approach

### Dataset

The project uses the **Canadian Vehicle Fuel Consumption Ratings Dataset** published by Natural Resources Canada (NRCan).

The dataset contains vehicle specifications and their corresponding fuel consumption and emissions measurements.

### Input Features

The model uses five primary features:

| Feature | Description |
|---|---|
| Engine Size (L) | Engine displacement in litres |
| Cylinders | Number of engine cylinders |
| Vehicle Class | Category of the vehicle |
| Transmission | Vehicle transmission type |
| Fuel Type | Type of fuel used |

**Prediction Target:** Combined Fuel Consumption (L/100 km).

### Data Preprocessing
- Removal of duplicate records.
- Separation of numerical and categorical features.
- One-Hot Encoding of categorical variables.
- Train-test splitting using an 80:20 ratio.

### Models Evaluated

Three regression algorithms were implemented and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The best-performing model is selected based on the lowest Root Mean Squared Error (RMSE) and saved as `model.pkl`.

### Model Performance

The following results are from the committed model evaluation:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | **0.466** | **0.656** | **0.950** |
| Gradient Boosting | 0.703 | 0.926 | 0.900 |
| Linear Regression | 0.748 | 0.984 | 0.887 |

**Random Forest achieved the best performance**, explaining approximately 95% of the variance in the test dataset.

---

## 📂 Project Structure

```text
Fuel-Efficiency-Predictor/
│
├── .streamlit/             # Streamlit configuration
│
├── data/
│   └── CO2_Emissions_Canada.csv
│
├── notebooks/              # Jupyter notebooks
│
├── app.py                  # Streamlit application
├── train.py                # Model training pipeline
├── model.pkl               # Trained ML model
│
├── metrics.csv             # Model evaluation metrics
├── importance.csv          # Feature importance results
├── test_eval.csv           # Test prediction results
│
├── fuelConsumption.zip     # Dataset archive
└── .gitignore
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/ShriyaSidana/Fuel-Efficiency-Predictor-.git
```

### 2. Navigate to the project directory

```bash
cd Fuel-Efficiency-Predictor-
```

### 3. Create a virtual environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### 5. Run the application

The repository already contains a trained model (`model.pkl`), so you can directly launch the application:

```bash
streamlit run app.py
```

The application will open in your default browser.

### Optional: Retrain the model

To train the models again and regenerate the evaluation files:

```bash
python train.py
```

This generates:
- `model.pkl`
- `metrics.csv`
- `importance.csv`
- `test_eval.csv`

---

## 📊 Model Evaluation

The application provides several visual tools to understand model performance:

- Model comparison charts.
- Actual vs. predicted consumption plots.
- Residual error distribution.
- Global permutation feature importance.
- Local feature contribution analysis.

These visualizations help evaluate predictive performance and understand how different vehicle characteristics influence fuel consumption estimates.

---

## ⚠️ Limitations

- Predictions are based on a limited set of five vehicle specifications.
- Vehicle weight, driving behaviour, road conditions, traffic, and weather are not directly considered.
- The underlying dataset represents standardized vehicle testing rather than actual individual driving conditions.
- Trip cost and emissions calculations are estimates and may differ from real-world values.
- Predictions for unusual vehicle configurations may be less reliable.

The results should therefore be treated as estimates rather than guaranteed real-world fuel consumption figures.

---

## 🚀 Future Scope

Potential improvements include:

- Integrating real-time fuel prices through external APIs.
- Incorporating additional vehicle specifications and real-world driving conditions.
- Expanding the dataset to include newer vehicle models and hybrid vehicles.
- Improving prediction accuracy through advanced ensemble techniques.
- Adding personalized vehicle recommendations based on fuel economy and ownership costs.
- Deploying the application for wider public access.
