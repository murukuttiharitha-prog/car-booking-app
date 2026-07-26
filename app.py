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
    df = pd.read_csv("cars.csv")
    df.columns = df.columns.str.strip()
    return df

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

# ---------- Simple login + signup ----------
# NOTE: this is a basic demo auth system (plain-text passwords, not secure for real production use)
USERS_FILE = "users.csv"
DEFAULT_USERS = {"demo": "demo123", "admin": "admin123"}

def load_users():
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        return dict(zip(df["username"], df["password"]))
    return {}

def save_user(username, password):
    users = load_users()
    users[username] = password
    df = pd.DataFrame(list(users.items()), columns=["username", "password"])
    df.to_csv(USERS_FILE, index=False)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Car Booking App")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    all_users = {**DEFAULT_USERS, **load_users()}

    with tab_login:
        st.caption("Demo credentials — username: `demo`  password: `demo123`")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if all_users.get(username) == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab_signup:
        new_username = st.text_input("Choose a username", key="signup_user")
        new_password = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account"):
            if not new_username or not new_password:
                st.error("Please enter both a username and password")
            elif new_username in all_users:
                st.error("That username is already taken")
            else:
                save_user(new_username, new_password)
                st.success("Account created! You can now log in from the Login tab.")

    st.stop()  # stops the rest of the app from running until logged in

# ---------- Logout button (shown once logged in) ----------
st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False
    st.rerun()

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

    # ---------- Filters ----------
    f1, f2, f3 = st.columns(3)
    with f1:
        type_filter = st.multiselect("Filter by type", options=cars_df["car_type"].unique())
    with f2:
        fuel_filter = st.multiselect("Filter by fuel", options=cars_df["fuel_type"].unique())
    with f3:
        max_price = st.slider("Max price per day (₹)",
                               min_value=int(cars_df["price_per_day"].min()),
                               max_value=int(cars_df["price_per_day"].max()),
                               value=int(cars_df["price_per_day"].max()))

    filtered_cars = cars_df.copy()
    if type_filter:
        filtered_cars = filtered_cars[filtered_cars["car_type"].isin(type_filter)]
    if fuel_filter:
        filtered_cars = filtered_cars[filtered_cars["fuel_type"].isin(fuel_filter)]
    filtered_cars = filtered_cars[filtered_cars["price_per_day"] <= max_price]

    st.caption(f"Showing {len(filtered_cars)} of {len(cars_df)} cars")

    # ---------- Car cards (visual grid instead of plain table) ----------
    CAR_COLORS = {"Hatchback": "#4F8EF7", "Sedan": "#2ECC71", "SUV": "#F39C12"}
    CAR_ICON = """
    <svg width="90" height="50" viewBox="0 0 90 50" xmlns="http://www.w3.org/2000/svg">
      <rect x="5" y="22" width="80" height="16" rx="4" fill="{color}"/>
      <path d="M18 22 L28 8 H62 L72 22 Z" fill="{color}"/>
      <path d="M30 21 L36 11 H54 L60 21 Z" fill="#EAF2FF"/>
      <circle cx="24" cy="40" r="7" fill="#222"/>
      <circle cx="24" cy="40" r="3" fill="#bbb"/>
      <circle cx="66" cy="40" r="7" fill="#222"/>
      <circle cx="66" cy="40" r="3" fill="#bbb"/>
    </svg>
    """

    n_cols = 4
    rows_of_cars = [filtered_cars.iloc[i:i+n_cols] for i in range(0, len(filtered_cars), n_cols)]
    for chunk in rows_of_cars:
        cols = st.columns(n_cols)
        for col, (_, car) in zip(cols, chunk.iterrows()):
            color = CAR_COLORS.get(car["car_type"], "#888")
            icon = CAR_ICON.format(color=color).replace("\n", "").strip()
            card_html = (
                f'<div style="border:1px solid #e0e0e0; border-radius:12px; padding:14px; text-align:center; margin-bottom:12px;">'
                f'{icon}'
                f'<div style="font-weight:600; margin-top:6px;">{car["car_name"]}</div>'
                f'<div style="color:#666; font-size:13px;">{car["car_type"]} · {car["seats"]} seats · {car["fuel_type"]}</div>'
                f'<div style="font-weight:600; margin-top:4px;">₹{car["price_per_day"]:,}/day</div>'
                f'</div>'
            )
            with col:
                st.markdown(card_html, unsafe_allow_html=True)
                )

    with st.expander("📋 View as table"):
        st.dataframe(filtered_cars, use_container_width=True)

    st.divider()
    st.header("📝 Book a Car & Predict Fare")

    col1, col2 = st.columns(2)

    with col1:
        car_name = st.selectbox("Choose a car", filtered_cars["car_name"] if len(filtered_cars) > 0 else cars_df["car_name"])
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

        st.session_state["last_prediction"] = {
            "car_name": car_name,
            "car_type": car_type,
            "days": days,
            "distance_km": distance_km,
            "is_weekend": int(is_weekend),
            "predicted_fare": round(float(predicted_fare), 2),
            "pickup_date": str(pickup_date),
        }

    # ---------- Step 2: Review trip details, then Book Now ----------
    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        st.divider()
        st.subheader("📋 Review Your Trip")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Car:** {pred['car_name']} ({pred['car_type']})")
            st.write(f"**Pickup date:** {pred['pickup_date']}")
            st.write(f"**Duration:** {pred['days']} day(s)")
        with c2:
            st.write(f"**Distance:** {pred['distance_km']} km")
            st.write(f"**Weekend trip:** {'Yes' if pred['is_weekend'] else 'No'}")
            st.metric("Estimated Fare", f"₹{pred['predicted_fare']:,.2f}")

        if st.button("🚗 Book Now", type="primary"):
            booking = pred.copy()
            booking["booking_id"] = len(load_bookings()) + 1
            booking["booked_by"] = st.session_state.username
            save_booking(booking)
            st.success(f"✅ Booking confirmed for {pred['car_name']}! Check 'Booking History' in the sidebar.")
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
