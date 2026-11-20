import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fuel Efficiency Predictor", page_icon="⛽", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.hero {background: linear-gradient(90deg, #0f766e, #2563eb); padding: 28px 32px;
       border-radius: 16px; margin-bottom: 20px;}
.hero h1 {margin: 0; font-size: 2.2rem; color: white;}
.hero p {margin: 6px 0 0; color: white; opacity: 0.9;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border-radius: 12px; padding: 14px 18px;}
.badge {display: inline-block; padding: 6px 16px; border-radius: 20px; color: white; font-weight: 600;}
</style>
<div class="hero">
  <h1>⛽ Fuel Efficiency Predictor</h1>
  <p>Predict fuel consumption, understand what drives it, and plan trip costs.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load model and data ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/CO2_Emissions_Canada.csv")

try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl not found. Run `python train.py` first.")
    st.stop()

df = load_data()
TARGET = "Fuel Consumption Comb (L/100 km)"


# ---------- Helpers ----------
def vehicle_inputs(key):
    c1, c2 = st.columns(2)
    vclass = c1.selectbox("Vehicle class", sorted(df["Vehicle Class"].unique()), key=key + "1")
    fuel = c2.selectbox("Fuel type", sorted(df["Fuel Type"].unique()), key=key + "2")
    engine = c1.slider("Engine size (L)", 0.8, 8.4, 2.0, 0.1, key=key + "3")
    cyl = c2.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12, 16], index=1, key=key + "4")
    trans = st.selectbox("Transmission", sorted(df["Transmission"].unique()), key=key + "5")

    if engine / cyl > 2.5:
        st.warning("Engine size looks too large for this number of cylinders. Please check.")
        return None

    return pd.DataFrame([{
        "Engine Size(L)": engine, "Cylinders": cyl,
        "Vehicle Class": vclass, "Transmission": trans, "Fuel Type": fuel,
    }])


def predict(X):
    try:
        return float(model.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def rating(pred):
    """Share of vehicles in the dataset that use MORE fuel than this one."""
    better_than = (df[TARGET] > pred).mean() * 100
    if better_than >= 75:
        return "Excellent", "#16a34a", better_than
    if better_than >= 50:
        return "Good", "#65a30d", better_than
    if better_than >= 25:
        return "Average", "#d97706", better_than
    return "Poor", "#dc2626", better_than


def gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": " L/100 km"},
        gauge={"axis": {"range": [3, 30]}, "bar": {"color": "#2563eb"},
               "steps": [{"range": [3, 8], "color": "#bbf7d0"},
                         {"range": [8, 12], "color": "#fef08a"},
                         {"range": [12, 30], "color": "#fecaca"}]}))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
    return fig


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "🧾 Trip Cost", "⚖️ Compare Vehicles", "📊 Model Insights"])

# ===== Tab 1: Prediction =====
with tab1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Vehicle details")
        X = vehicle_inputs("p")
        go_predict = st.button("Predict", type="primary", key="btn_predict")

    with right:
        if X is not None and go_predict:
            pred = predict(X)
            if pred:
                label, color, pct = rating(pred)
                st.plotly_chart(gauge(pred))
                st.markdown(
                    f'<span class="badge" style="background:{color}">{label}</span> '
                    f'&nbsp; More efficient than **{pct:.0f}%** of vehicles in the dataset.',
                    unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Fuel consumption", f"{pred:.2f} L/100 km")
                m2.metric("Mileage", f"{100 / pred:.1f} km/L")

                # What-if analysis
                st.markdown("#### What if the engine size changed?")
                sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
                rows = pd.concat([X] * len(sizes), ignore_index=True)
                rows["Engine Size(L)"] = sizes
                wi = pd.DataFrame({"Engine size (L)": sizes,
                                   "L/100 km": model.predict(rows)})
                fig = px.line(wi, x="Engine size (L)", y="L/100 km", markers=True)
                fig.add_scatter(x=[X["Engine Size(L)"][0]], y=[pred], mode="markers",
                                marker=dict(size=14, color="red"), name="Your input")
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig)
        else:
            st.info("Enter vehicle details and click **Predict**.")

# ===== Tab 2: Trip Cost =====
with tab2:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Vehicle and trip")
        X = vehicle_inputs("t")
        distance = st.number_input("Trip distance (km)", 1.0, 20000.0, 300.0, 10.0)
        price = st.number_input("Fuel price per litre (₹)", 1.0, 500.0, 100.0, 1.0)
        go_cost = st.button("Calculate", type="primary", key="btn_cost")

    with right:
        if X is not None and go_cost:
            pred = predict(X)
            if pred:
                litres = pred * distance / 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Consumption", f"{pred:.2f} L/100 km")
                c2.metric("Fuel needed", f"{litres:.1f} L")
                c3.metric("Trip cost", f"₹{litres * price:,.0f}")

                st.markdown("#### Cost vs distance")
                dists = list(range(0, int(distance * 2) + 1, max(1, int(distance // 10))))
                cost_df = pd.DataFrame({"Distance (km)": dists,
                                        "Cost (₹)": [pred * d / 100 * price for d in dists]})
                fig = px.area(cost_df, x="Distance (km)", y="Cost (₹)")
                fig.add_vline(x=distance, line_dash="dash", line_color="red")
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig)
        else:
            st.info("Fill in the trip details and click **Calculate**.")

# ===== Tab 3: Compare =====
with tab3:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🚗 Vehicle A")
        XA = vehicle_inputs("a")
    with col2:
        st.subheader("🚙 Vehicle B")
        XB = vehicle_inputs("b")

    d1, d2 = st.columns(2)
    distance = d1.number_input("Distance per year (km)", 1000.0, 100000.0, 15000.0, 1000.0)
    price = d2.number_input("Fuel price per litre (₹)", 1.0, 500.0, 100.0, 1.0, key="cmp_price")

    if XA is not None and XB is not None and st.button("Compare", type="primary", key="btn_cmp"):
        a, b = predict(XA), predict(XB)
        if a and b:
            cost_a, cost_b = a * distance / 100 * price, b * distance / 100 * price
            m1, m2, m3 = st.columns(3)
            m1.metric("Vehicle A", f"{a:.2f} L/100 km", f"₹{cost_a:,.0f}/yr", delta_color="off")
            m2.metric("Vehicle B", f"{b:.2f} L/100 km", f"₹{cost_b:,.0f}/yr", delta_color="off")
            winner = "A" if cost_a < cost_b else "B"
            m3.metric("Cheaper to run", f"Vehicle {winner}", f"saves ₹{abs(cost_a - cost_b):,.0f}/yr")

            fig = px.bar(x=["Vehicle A", "Vehicle B"], y=[cost_a, cost_b],
                         color=["Vehicle A", "Vehicle B"],
                         labels={"x": "", "y": "Yearly fuel cost (₹)"})
            fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig)

# ===== Tab 4: Model Insights =====
with tab4:
    st.subheader("Model comparison")
    try:
        st.dataframe(pd.read_csv("metrics.csv"), hide_index=True)
        st.caption("Lower MAE and RMSE are better. Higher R² is better. "
                   "The best model (lowest RMSE) is the one used in this app.")
    except FileNotFoundError:
        st.info("Run train.py to generate metrics.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What drives fuel consumption?")
        try:
            imp = pd.read_csv("importance.csv").sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig)
        except FileNotFoundError:
            st.info("Run train.py to generate feature importance.")
    with right:
        st.subheader("Engine size vs consumption")
        fig = px.scatter(df.sample(1500, random_state=1), x="Engine Size(L)", y=TARGET,
                         color="Fuel Type", opacity=0.6)
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig)