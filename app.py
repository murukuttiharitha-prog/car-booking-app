python
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Booking App", page_icon="🚗", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_cars():
    return pd.read_csv("cars.csv")

# ---------- Generate synthetic training data + train model (in-app, no pkl files needed) ----------
@st.cache_resource
def train_fare_model():
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

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, encoder

cars_df = load_cars()
model, encoder = train_fare_model()

BOOKINGS_FILE = "bookings.csv"

def load_bookings():
    if os.path.exists(BOOKINGS_FILE):
        return pd.read_csv(BOOKINGS_FILE)
    return pd.DataFrame(columns=[
        "booking_id", "car_name", "car_type", "days", "distance_km",
        "is_weekend", "predicted_fare", "pickup_date"
    ])

def save_booking(row):
    bookings = load_bookings()
    bookings = pd.concat([bookings, pd.DataFrame([row])], ignore_index=True)
    bookings.to_csv(BOOKINGS_FILE, index=False)

# ---------- Sidebar navigation ----------
st.sidebar.title("🚗 Car Booking App")
page = st.sidebar.radio("Go to", ["Browse & Book", "Booking History"])

# ---------- Page 1: Browse & Book ----------
if page == "Browse & Book":
    st.title("🚗 Available Cars")
    st.dataframe(cars_df, use_container_width=True)

    st.divider()
    st.header("📝 Book a Car & Predict Fare")

    col1, col2 = st.columns(2)

    with col1:
        car_name = st.selectbox("Choose a car", cars_df["car_name"])
        car_row = cars_df[cars_df["car_name"] == car_name].iloc[0]
        car_type = car_row["car_type"]
        st.write(f"**Type:** {car_type} | **Seats:** {car_row['seats']} | **Fuel:** {car_row['fuel_type']}")

        pickup_date = st.date_input("Pickup date", date.today())
        days = st.number_input("Number of days", min_value=1, max_value=30, value=3)

    with col2:
        distance_km = st.number_input("Estimated distance (km)", min_value=1, max_value=3000, value=150)
        is_weekend = st.checkbox("Trip includes a weekend?")

    if st.button("🔮 Predict Fare", type="primary"):
        car_type_encoded = encoder.transform([car_type])[0]
        features = np.array([[car_type_encoded, days, distance_km, int(is_weekend)]])
        predicted_fare = model.predict(features)[0]

        st.success(f"Estimated Fare: ₹{predicted_fare:,.2f}")
        st.session_state["last_prediction"] = {
            "car_name": car_name,
            "car_type": car_type,
            "days": days,
            "distance_km": distance_km,
            "is_weekend": int(is_weekend),
            "predicted_fare": round(float(predicted_fare), 2),
            "pickup_date": str(pickup_date),
        }

    if "last_prediction" in st.session_state:
        st.info(f"Ready to book **{st.session_state['last_prediction']['car_name']}** "
                f"for ₹{st.session_state['last_prediction']['predicted_fare']:,.2f}")
        if st.button("✅ Confirm Booking"):
            booking = st.session_state["last_prediction"].copy()
            booking["booking_id"] = len(load_bookings()) + 1
            save_booking(booking)
            st.success("Booking confirmed! Check 'Booking History' in the sidebar.")
            del st.session_state["last_prediction"]

# ---------- Page 2: Booking History ----------
elif page == "Booking History":
    st.title("📊 Booking History")
    bookings = load_bookings()

    if bookings.empty:
        st.info("No bookings yet. Go to 'Browse & Book' to make one.")
    else:
        st.dataframe(bookings, use_container_width=True)

        st.subheader("Revenue by Car Type")
        revenue_by_type = bookings.groupby("car_type")["predicted_fare"].sum()
        st.bar_chart(revenue_by_type)

        st.subheader("Bookings Over Time")
        bookings_by_date = bookings.groupby("pickup_date")["booking_id"].count()
        st.line_chart(bookings_by_date)
