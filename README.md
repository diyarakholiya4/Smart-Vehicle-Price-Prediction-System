# 🚗 Smart Vehicle Price Prediction & Recommendation System

A Machine Learning based web application developed using **Python** and **Streamlit** that predicts the selling price of cars and bikes, recommends similar vehicles, compares vehicles, and provides an interactive analytics dashboard.

---

## 📌 Project Overview

The Smart Vehicle Price Prediction & Recommendation System is designed to help users make informed decisions while buying or selling vehicles.

The application uses trained Machine Learning models to predict vehicle prices and provides intelligent recommendations based on similar vehicle specifications.

It also includes comparison tools, ownership cost estimation, and an interactive dashboard for vehicle analytics.

---

## ✨ Features

### 🚗 Car Module

- Predict Car Selling Price
- Recommend Similar Cars
- Compare Two Cars

### 🏍️ Bike Module

- Predict Bike Price
- Recommend Similar Bikes
- Compare Two Bikes

### 📊 Dashboard

- Vehicle Statistics
- Car Fuel Distribution
- Cars by Manufacturing Year
- Bike Company Distribution
- Bike Fuel Distribution
- Engine Size Distribution
- Selling Price vs KM Driven
- Dataset Preview

### 💰 Ownership Cost Calculator

Estimate total ownership cost based on:

- Purchase Price
- Fuel Cost
- Maintenance Cost
- Insurance Cost
- Ownership Years

### ℹ️ About Page

Project details and technologies used.

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly

---

## 🤖 Machine Learning Models

### Car Price Prediction

Model trained using vehicle features:

- Year
- KM Driven
- Fuel Type
- Seller Type
- Transmission
- Owner

### Bike Price Prediction

Model trained using:

- Company
- Bike Name
- Mileage
- Engine Capacity
- Fuel Type

---

## 📂 Project Structure

```
Smart-Vehicle-Price-Prediction-System/

│── app.py
│── README.md
│── requirements.txt

├── datasets/
│   ├── Bikes.csv
│   ├── bike_preprocessed_data.csv
│   └── preprocessed_car_data.csv

├── models/
│   ├── best_model.joblib
│   ├── bike_best_model.joblib
│   ├── bike_label_encoder.joblib
│   └── label_encoder.joblib

├── assets/

└── utils/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/diyarakholiya4/Smart-Vehicle-Price-Prediction-System.git
```

Go to the project folder

```bash
cd Smart-Vehicle-Price-Prediction-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Dashboard

The dashboard provides interactive visualizations including:

- Car Fuel Distribution
- Cars by Manufacturing Year
- Bike Company Distribution
- Bike Fuel Distribution
- Engine Size Distribution
- Selling Price vs KM Driven
- Vehicle Statistics

---

## 📸 Screenshots

Add screenshots of:

- Home Page
- Car Prediction
- Bike Prediction
- Dashboard
- Car Comparison
- Bike Comparison

---

## 🎯 Future Enhancements

- User Login System
- Database Integration
- Real-time Vehicle Data
- Advanced Recommendation Engine
- AI-based Price Insights
- Cloud Deployment
- Mobile Responsive Interface

---

## 👩‍💻 Developer

**Diya Rakholiya**

Machine Learning & Python Developer

---

## 📜 License

This project is developed for educational and internship purposes.

---

## ⭐ Acknowledgement

This project was developed as part of an internship and academic learning in Machine Learning, Data Analytics, and Web Application Development using Streamlit.