import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Smart Vehicle Price Prediction & Recommendation System",
    page_icon="🚗",
    layout="wide"
)
# ======================================================
# CUSTOM CSS
# ======================================================

def load_css():
    st.markdown("""
    <style>

    .main{
        background-color:#F8F9FA;
    }

    .stButton>button{
        width:100%;
        background:#0E6EFD;
        color:white;
        border-radius:10px;
        height:45px;
        font-size:17px;
        border:none;
    }

    .stButton>button:hover{
        background:#084298;
        color:white;
    }

    div[data-testid="metric-container"]{
        border-radius:15px;
        padding:15px;
        box-shadow:0px 0px 8px rgba(0,0,0,0.15);
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# ======================================================
# LOAD MODELS
# ======================================================

@st.cache_resource
def load_models():

    car_model = joblib.load("models/best_model.joblib")

    bike_model = joblib.load("models/bike_best_model.joblib")

    bike_encoder = joblib.load("models/bike_label_encoder.joblib")

    return car_model, bike_model, bike_encoder


best_model, bike_model, bike_label_encoder = load_models()

# ======================================================
# LOAD DATASETS
# ======================================================

@st.cache_data
def load_data():

    car_df = pd.read_csv("datasets/preprocessed_car_data.csv")

    bike_df = pd.read_csv("datasets/Bikes.csv")

    return car_df, bike_df


car_df, bike_df = load_data()

# ======================================================
# CLEAN BIKE DATA
# ======================================================

if "Unnamed: 0" in bike_df.columns:
    bike_df = bike_df.drop(columns=["Unnamed: 0"])

bike_df["Milage (in KMPL)"] = bike_df["Milage (in KMPL)"].replace("-", np.nan)
bike_df["Engine (in cc)"] = bike_df["Engine (in cc)"].replace("-", np.nan)

bike_df["Milage (in KMPL)"] = pd.to_numeric(
    bike_df["Milage (in KMPL)"],
    errors="coerce"
)

bike_df["Engine (in cc)"] = pd.to_numeric(
    bike_df["Engine (in cc)"],
    errors="coerce"
)

bike_df["Milage (in KMPL)"] = bike_df["Milage (in KMPL)"].fillna(
    bike_df["Milage (in KMPL)"].median()
)

bike_df["Engine (in cc)"] = bike_df["Engine (in cc)"].fillna(
    bike_df["Engine (in cc)"].median()
)

# ======================================================
# CAR RECOMMENDATION FUNCTION
# ======================================================

def recommend_cars(
    df,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner
):

    temp = df.copy()

    temp = temp[
        (temp["year"] >= year - 2) &
        (temp["year"] <= year + 2)
    ]

    temp = temp[
        (temp["km_driven"] >= km_driven - 10000) &
        (temp["km_driven"] <= km_driven + 10000)
    ]

    temp = temp[temp["fuel"] == fuel]
    temp = temp[temp["seller_type"] == seller_type]
    temp = temp[temp["transmission"] == transmission]
    temp = temp[temp["owner"] == owner]

    if temp.empty:
        return temp

    temp["Score"] = abs(temp["km_driven"] - km_driven)

    return temp.sort_values("Score").head(10)

# ======================================================
# BIKE RECOMMENDATION FUNCTION
# ======================================================

def recommend_bikes(
    df,
    company,
    bike_name,
    fuel_type,
    engine,
    mileage
):

    temp = df[
        (df["Company"] == company) &
        (df["Bike_Name"] != bike_name) &
        (df["Fuel Type"] == fuel_type) &
        (abs(df["Engine (in cc)"] - engine) <= 20) &
        (abs(df["Milage (in KMPL)"] - mileage) <= 10)
    ]

    return temp

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🚗 Smart Vehicle System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚗 Predict Car Price",
        "🚗 Recommend Cars",
        "🚗 Compare Cars",
        "🏍️ Predict Bike Price",
        "🏍️ Recommend Bikes",
        "🏍️ Compare Bikes",
        "📊 Dashboard",
        "💰 Ownership Cost",
        "ℹ️ About"
    ]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.title("🚗 Smart Vehicle Price Prediction & Recommendation System")

    st.markdown("""
    ### Welcome

    This application helps users predict vehicle prices,
    compare vehicles, and receive recommendations using
    Machine Learning.
    """)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vehicle Types", "2")
    c2.metric("Prediction Models", "2")
    c3.metric("Recommendation Systems", "2")
    c4.metric("Accuracy", "95%+")

    st.divider()

    st.subheader("✨ Features")

    left, right = st.columns(2)

    with left:
        st.success("🚗 Car Price Prediction")
        st.success("🚗 Car Recommendation")
        st.success("🚗 Car Comparison")
        st.success("🏍 Bike Price Prediction")

    with right:
        st.success("🏍 Bike Recommendation")
        st.success("🏍 Bike Comparison")
        st.success("📊 Dashboard")
        st.success("💰 Ownership Cost Calculator")

    st.divider()

    st.subheader("⚙ Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Machine Learning
    - Scikit-Learn
    - Pandas
    - Joblib
    """)

    st.divider()

    st.subheader("📌 Project Workflow")

    st.info("""
    User Input
        ↓
    Data Preprocessing
        ↓
    Machine Learning Model
        ↓
    Price Prediction / Recommendation
        ↓
    Result Display
    """)

# ======================================================
# CAR PRICE PREDICTION
# ======================================================

elif page == "🚗 Predict Car Price":

    st.header("🚗 Predict Car Price")

    year = st.slider(
        "Manufacturing Year",
        1990,
        2024,
        2018
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        0,
        500000,
        50000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

    seller = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner"
        ]
    )

    fuel_map = {
        "CNG": 0,
        "Diesel": 1,
        "Electric": 2,
        "LPG": 3,
        "Petrol": 4
    }

    seller_map = {
        "Dealer": 0,
        "Individual": 1,
        "Trustmark Dealer": 2
    }

    transmission_map = {
        "Automatic": 0,
        "Manual": 1
    }

    owner_map = {
        "First Owner": 0,
        "Second Owner": 2,
        "Third Owner": 4
    }

    if st.button("Predict Car Price"):

        input_df = pd.DataFrame(
            [[
                year,
                km_driven,
                fuel_map[fuel],
                seller_map[seller],
                transmission_map[transmission],
                owner_map[owner]
            ]],
            columns=[
                "year",
                "km_driven",
                "fuel",
                "seller_type",
                "transmission",
                "owner"
            ]
        )

        prediction = best_model.predict(input_df)[0]

        st.success(
            f"Estimated Selling Price : ₹ {prediction:,.2f}"
        )

# ======================================================
# CAR RECOMMENDATION
# ======================================================

elif page == "🚗 Recommend Cars":

    st.header("🚗 Recommend Similar Cars")

    year = st.slider(
        "Year",
        1990,
        2024,
        2018
    )

    km = st.number_input(
        "KM Driven",
        min_value=0,
        max_value=500000,
        value=50000
    )

    fuel = st.selectbox(
        "Fuel",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

    seller = st.selectbox(
        "Seller",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner"
        ]
    )

    fuel_map = {
        "CNG": 0,
        "Diesel": 1,
        "Electric": 2,
        "LPG": 3,
        "Petrol": 4
    }

    seller_map = {
        "Dealer": 0,
        "Individual": 1,
        "Trustmark Dealer": 2
    }

    transmission_map = {
        "Automatic": 0,
        "Manual": 1
    }

    owner_map = {
        "First Owner": 0,
        "Second Owner": 2,
        "Third Owner": 4
    }

    if st.button("Recommend Cars"):

        result = recommend_cars(
            car_df,
            year,
            km,
            fuel_map[fuel],
            seller_map[seller],
            transmission_map[transmission],
            owner_map[owner]
        )

        if result.empty:
            st.warning("No matching cars found.")

        else:
            st.success(f"{len(result)} Cars Found")

            st.dataframe(
                result[
                    [
                        "name",
                        "year",
                        "selling_price",
                        "km_driven"
                    ]
                ],
                use_container_width=True
            )
# ======================================================
# BIKE PRICE PREDICTION
# ======================================================

elif page == "🏍️ Predict Bike Price":

    st.header("🏍️ Predict Bike Price")

    company = st.selectbox(
        "Company",
        sorted(bike_df["Company"].unique())
    )

    bike_name = st.selectbox(
        "Bike Name",
        sorted(
            bike_df[
                bike_df["Company"] == company
            ]["Bike_Name"].unique()
        )
    )

    fuel = st.selectbox(
        "Fuel Type",
        sorted(bike_df["Fuel Type"].unique())
    )

    mileage = st.number_input(
        "Mileage (KMPL)",
        0.0,
        100.0,
        50.0
    )

    engine = st.number_input(
        "Engine (CC)",
        0.0,
        1000.0,
        125.0
    )

    if st.button("Predict Bike Price"):

        input_df = pd.DataFrame(
            [[
                company,
                bike_name,
                mileage,
                engine,
                fuel
            ]],
            columns=[
                "Company",
                "Bike_Name",
                "Milage (in KMPL)",
                "Engine (in cc)",
                "Fuel Type"
            ]
        )

        input_df["Company"] = bike_label_encoder["Company"].transform(
            input_df["Company"]
        )

        input_df["Bike_Name"] = bike_label_encoder["Bike_Name"].transform(
            input_df["Bike_Name"]
        )

        input_df["Fuel Type"] = bike_label_encoder["Fuel Type"].transform(
            input_df["Fuel Type"]
        )

        prediction = bike_model.predict(input_df)[0]

        st.success(f"Estimated Bike Price : ₹ {prediction:,.2f}")

# ======================================================
# BIKE RECOMMENDATION
# ======================================================

elif page == "🏍️ Recommend Bikes":

    st.header("🏍️ Recommend Similar Bikes")

    company = st.selectbox(
        "Company",
        sorted(bike_df["Company"].unique())
    )

    bike_name = st.selectbox(
        "Bike",
        sorted(
            bike_df[
                bike_df["Company"] == company
            ]["Bike_Name"].unique()
        )
    )

    selected = bike_df[
        (bike_df["Company"] == company) &
        (bike_df["Bike_Name"] == bike_name)
    ].iloc[0]

    fuel = selected["Fuel Type"]
    mileage = selected["Milage (in KMPL)"]
    engine = selected["Engine (in cc)"]

    st.write(f"Fuel Type : {fuel}")
    st.write(f"Mileage : {mileage}")
    st.write(f"Engine : {engine} CC")

    if st.button("Recommend Bikes"):

        result = recommend_bikes(
            bike_df,
            company,
            bike_name,
            fuel,
            engine,
            mileage
        )

        if result.empty:
            st.warning("No similar bikes found.")

        else:
            st.success(f"{len(result)} Similar Bikes Found")

            st.dataframe(
                result[
                    [
                        "Company",
                        "Bike_Name",
                        "Price",
                        "Milage (in KMPL)",
                        "Engine (in cc)"
                    ]
                ]
            )

# ======================================================
# CAR COMPARISON
elif page == "🚗 Compare Cars":

    st.header("🚗 Compare Cars")

    car_names = sorted(car_df["name"].unique())

    col1, col2 = st.columns(2)

    with col1:
        car1 = st.selectbox(
            "Select Car 1",
            car_names,
            key="car1"
        )

    with col2:
        car2 = st.selectbox(
            "Select Car 2",
            car_names,
            index=1 if len(car_names) > 1 else 0,
            key="car2"
        )

    if st.button("Compare Cars"):

        car1_data = car_df[car_df["name"] == car1].iloc[0]
        car2_data = car_df[car_df["name"] == car2].iloc[0]

        comparison = pd.DataFrame({
            "Feature": [
                "Year",
                "Selling Price",
                "KM Driven",
                "Fuel",
                "Seller Type",
                "Transmission",
                "Owner"
            ],
            car1: [
                car1_data["year"],
                car1_data["selling_price"],
                car1_data["km_driven"],
                car1_data["fuel"],
                car1_data["seller_type"],
                car1_data["transmission"],
                car1_data["owner"]
            ],
            car2: [
                car2_data["year"],
                car2_data["selling_price"],
                car2_data["km_driven"],
                car2_data["fuel"],
                car2_data["seller_type"],
                car2_data["transmission"],
                car2_data["owner"]
            ]
        })

        st.subheader("Comparison Result")
        st.dataframe(comparison, use_container_width=True)

# ======================================================
# BIKE COMPARISON
# ======================================================

elif page == "🏍️ Compare Bikes":

    st.header("🏍️ Compare Bikes")

    bike_names = sorted(bike_df["Bike_Name"].unique())

    col1, col2 = st.columns(2)

    with col1:
        bike1 = st.selectbox(
            "Select Bike 1",
            bike_names,
            key="bike1"
        )

    with col2:
        bike2 = st.selectbox(
            "Select Bike 2",
            bike_names,
            index=1 if len(bike_names) > 1 else 0,
            key="bike2"
        )

    if st.button("Compare Bikes"):

        bike1_data = bike_df[bike_df["Bike_Name"] == bike1].iloc[0]
        bike2_data = bike_df[bike_df["Bike_Name"] == bike2].iloc[0]

        comparison = pd.DataFrame({
            "Feature": [
                "Company",
                "Price",
                "Mileage",
                "Engine",
                "Fuel Type"
            ],
            bike1: [
                bike1_data["Company"],
                bike1_data["Price"],
                bike1_data["Milage (in KMPL)"],
                bike1_data["Engine (in cc)"],
                bike1_data["Fuel Type"]
            ],
            bike2: [
                bike2_data["Company"],
                bike2_data["Price"],
                bike2_data["Milage (in KMPL)"],
                bike2_data["Engine (in cc)"],
                bike2_data["Fuel Type"]
            ]
        })

        st.success("Comparison Completed")

        st.dataframe(
            comparison,
            use_container_width=True
        )
# ======================================================
# DASHBOARD
# ======================================================

elif page == "📊 Dashboard":

    st.title("📊 Smart Vehicle Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🚗 Cars", len(car_df))
    c2.metric("🏍 Bikes", len(bike_df))
    c3.metric("🤖 ML Models", "2")
    c4.metric("📈 Accuracy", "95%+")

    st.markdown("---")

    row1_col1, row1_col2 = st.columns(2)

    # ---------------- CAR FUEL ----------------
    with row1_col1:

        fuel = (
            car_df["fuel"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        fuel.columns = ["Fuel", "Count"]

        fig = px.pie(
            fuel,
            names="Fuel",
            values="Count",
            title="🚗 Car Fuel Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CAR YEAR ----------------
    with row1_col2:

        year = (
            car_df["year"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        year.columns = ["Year", "Cars"]

        fig = px.bar(
            year,
            x="Year",
            y="Cars",
            title="📅 Cars by Manufacturing Year"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    row2_col1, row2_col2 = st.columns(2)

    # ---------------- BIKE COMPANY ----------------
    with row2_col1:

        company = (
            bike_df["Company"]
            .value_counts()
            .reset_index()
        )

        company.columns = ["Company", "Count"]

        fig = px.bar(
            company,
            x="Company",
            y="Count",
            title="🏍 Bike Company Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- BIKE FUEL ----------------
    with row2_col2:

        bikefuel = (
            bike_df["Fuel Type"]
            .value_counts()
            .reset_index()
        )

        bikefuel.columns = ["Fuel", "Count"]

        fig = px.pie(
            bikefuel,
            names="Fuel",
            values="Count",
            title="⛽ Bike Fuel Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    row3_col1, row3_col2 = st.columns(2)

    # ---------------- ENGINE ----------------
    with row3_col1:

        fig = px.histogram(
            bike_df,
            x="Engine (in cc)",
            nbins=20,
            title="⚙ Engine Size Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CAR PRICE ----------------
    with row3_col2:

        fig = px.scatter(
            car_df,
            x="km_driven",
            y="selling_price",
            color="fuel",
            title="💰 Selling Price vs KM Driven"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🚗 Car Dataset Preview")
    st.dataframe(car_df.head(), use_container_width=True)

    st.subheader("🏍 Bike Dataset Preview")
    st.dataframe(
        bike_df[
            [
                "Company",
                "Bike_Name",
                "Price",
                "Milage (in KMPL)",
                "Engine (in cc)"
            ]
        ].head(),
        use_container_width=True
    )
# ======================================================
# OWNERSHIP COST
# ======================================================

elif page == "💰 Ownership Cost":

    st.header("💰 Ownership Cost Calculator")

    purchase_price = st.number_input(
        "Vehicle Purchase Price (₹)",
        min_value=0,
        value=500000
    )

    fuel_cost = st.number_input(
        "Annual Fuel Cost (₹)",
        min_value=0,
        value=50000
    )

    maintenance = st.number_input(
        "Annual Maintenance Cost (₹)",
        min_value=0,
        value=10000
    )

    insurance = st.number_input(
        "Annual Insurance Cost (₹)",
        min_value=0,
        value=8000
    )

    years = st.slider(
        "Ownership Years",
        1,
        10,
        5
    )

    if st.button("Calculate Total Cost"):

        total = purchase_price + years * (
            fuel_cost +
            maintenance +
            insurance
        )

        st.success(f"Estimated Ownership Cost: ₹ {total:,.2f}")

# ======================================================
# ABOUT
# ======================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
# Smart Vehicle Price Prediction & Recommendation System


### Technologies Used

- Python
- Streamlit
- Machine Learning
- Scikit-Learn
- Pandas
- NumPy
- Joblib

### Features

✅ Car Price Prediction

✅ Bike Price Prediction

✅ Car Recommendation

✅ Bike Recommendation

✅ Car Comparison

✅ Bike Comparison

✅ Dashboard

✅ Ownership Cost Calculator

### Description

This application helps users predict the selling price of cars and bikes
using Machine Learning models. It also recommends similar vehicles
based on user requirements and provides a simple ownership cost
calculator with an interactive dashboard.

""")

st.sidebar.markdown("---")


st.markdown("---")
st.caption("© 2026 | Smart Vehicle Price Prediction & Recommendation System ")