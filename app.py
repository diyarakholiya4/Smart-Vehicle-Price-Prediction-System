import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD MODELS --------------------
best_model = joblib.load('best_model.joblib')
df_preprocessed = pd.read_csv('preprocessed_car_data.csv')

bike_model = joblib.load("bike_best_model.joblib")
bike_label_encoder = joblib.load("bike_label_encoder.joblib")

bike_df = pd.read_csv("Bikes.csv")
bike_df = bike_df.drop(columns=["Unnamed: 0"])

# -------------------- BIKE CLEANING --------------------
bike_df["Milage (in KMPL)"] = bike_df["Milage (in KMPL)"].replace("-", pd.NA)
bike_df["Engine (in cc)"] = bike_df["Engine (in cc)"].replace("-", pd.NA)

bike_df["Milage (in KMPL)"] = pd.to_numeric(bike_df["Milage (in KMPL)"])
bike_df["Engine (in cc)"] = pd.to_numeric(bike_df["Engine (in cc)"])

bike_df["Milage (in KMPL)"] = bike_df["Milage (in KMPL)"].fillna(
    bike_df["Milage (in KMPL)"].median()
)

bike_df["Engine (in cc)"] = bike_df["Engine (in cc)"].fillna(
    bike_df["Engine (in cc)"].median()
)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(layout='wide', page_title='Vehicle System')

st.title("🚗 Smart Vehicle Price Prediction & Recommendation System")

# -------------------- SIDEBAR --------------------
st.sidebar.title("Navigation")

vehicle = st.sidebar.selectbox(
    "Select Vehicle",
    ["🚗 Car", "🏍️ Bike"]
)

if vehicle == "🚗 Car":
    page = st.sidebar.radio("Go to", ["Predict Car Price", "Recommend Cars"])
else:
    page = st.sidebar.radio("Go to", ["Predict Bike Price", "Recommend Bikes"])

# -------------------- CAR RECOMMENDATION --------------------
def recommend_cars(df, year, km_driven, fuel, seller_type, transmission, owner):
    temp = df.copy()

    temp = temp[
        (temp['year'] >= year - 2) &
        (temp['year'] <= year + 2)
    ]

    temp = temp[
        (temp['km_driven'] >= km_driven - 10000) &
        (temp['km_driven'] <= km_driven + 10000)
    ]

    temp = temp[temp['fuel'] == fuel]
    temp = temp[temp['seller_type'] == seller_type]
    temp = temp[temp['transmission'] == transmission]
    temp = temp[temp['owner'] == owner]

    temp['score'] = abs(temp['km_driven'] - km_driven)

    return temp.sort_values('score').head(10)

# -------------------- BIKE RECOMMENDATION --------------------
def recommend_bikes(df, company, bike_name, fuel_type, engine, mileage):

    temp = df[
        (df["Company"] == company) &
        (df["Bike_Name"] != bike_name) &
        (df["Fuel Type"] == fuel_type) &
        (abs(df["Engine (in cc)"] - engine) <= 20) &
        (abs(df["Milage (in KMPL)"] - mileage) <= 10)
    ]

    return temp

# ==================== CAR PREDICTION ====================
if page == "Predict Car Price":

    st.header("🚗 Predict Car Price")

    year = st.slider("Year", 1990, 2024, 2015)
    km_driven = st.number_input("KM Driven", 0, 500000, 50000)

    fuel = st.selectbox("Fuel", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller = st.selectbox("Seller", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner'])

    fuel_map = {'Petrol':4,'Diesel':1,'CNG':0,'LPG':3,'Electric':2}
    seller_map = {'Individual':1,'Dealer':0,'Trustmark Dealer':2}
    trans_map = {'Manual':1,'Automatic':0}
    owner_map = {'First Owner':0,'Second Owner':2,'Third Owner':4}

    if st.button("Predict Car Price"):
        input_df = pd.DataFrame([[year, km_driven,
                                 fuel_map[fuel],
                                 seller_map[seller],
                                 trans_map[transmission],
                                 owner_map[owner]]],
                                columns=['year','km_driven','fuel','seller_type','transmission','owner'])

        pred = best_model.predict(input_df)[0]
        st.success(f"Predicted Price: ₹ {pred:,.2f}")

# ==================== CAR RECOMMEND ====================
elif page == "Recommend Cars":

    st.header("🚗 Recommend Cars")

    rec_year = st.slider("Year", 1990, 2024, 2015)
    rec_km = st.number_input("KM Driven", 0, 500000, 60000)

    fuel = st.selectbox("Fuel", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller = st.selectbox("Seller", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner'])

    fuel_map = {'Petrol':4,'Diesel':1,'CNG':0,'LPG':3,'Electric':2}
    seller_map = {'Individual':1,'Dealer':0,'Trustmark Dealer':2}
    trans_map = {'Manual':1,'Automatic':0}
    owner_map = {'First Owner':0,'Second Owner':2,'Third Owner':4}

    if st.button("Get Cars"):
        result = recommend_cars(
            df_preprocessed,
            rec_year,
            rec_km,
            fuel_map[fuel],
            seller_map[seller],
            trans_map[transmission],
            owner_map[owner]
        )

        st.dataframe(result)

# ==================== BIKE PREDICTION ====================
elif page == "Predict Bike Price":

    st.header("🏍️ Predict Bike Price")

    company = st.selectbox("Company", sorted(bike_df["Company"].unique()))
    bike_name = st.selectbox("Bike", sorted(bike_df[bike_df["Company"]==company]["Bike_Name"].unique()))
    fuel = st.selectbox("Fuel", sorted(bike_df["Fuel Type"].unique()))

    mileage = st.number_input("Mileage", 0.0, 100.0, 50.0)
    engine = st.number_input("Engine", 0.0, 500.0, 125.0)

    if st.button("Predict Bike Price"):

        input_df = pd.DataFrame([[company, bike_name, mileage, engine, fuel]],
                                columns=["Company","Bike_Name","Milage (in KMPL)","Engine (in cc)","Fuel Type"])

        for col in ["Company","Bike_Name","Fuel Type"]:
            input_df[col] = bike_label_encoder[col].transform(input_df[col])

        pred = bike_model.predict(input_df)[0]
        st.success(f"Predicted Bike Price: ₹ {pred:,.2f}")

# ==================== BIKE RECOMMEND ====================
elif page == "Recommend Bikes":

    st.header("🏍️ Recommend Bikes")

    company = st.selectbox("Company", sorted(bike_df["Company"].unique()))
    bike_name = st.selectbox("Bike", sorted(bike_df[bike_df["Company"]==company]["Bike_Name"].unique()))

    selected = bike_df[
        (bike_df["Company"] == company) &
        (bike_df["Bike_Name"] == bike_name)
    ].iloc[0]

    fuel = selected["Fuel Type"]
    mileage = selected["Milage (in KMPL)"]
    engine = selected["Engine (in cc)"]

    st.write(f"Fuel: {fuel}")
    st.write(f"Mileage: {mileage}")
    st.write(f"Engine: {engine}")

    if st.button("Recommend Bikes"):

        result = recommend_bikes(
            bike_df,
            company,
            bike_name,
            fuel,
            engine,
            mileage
        )

        st.dataframe(result)
