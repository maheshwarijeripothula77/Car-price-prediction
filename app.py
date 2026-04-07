import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.markdown(
    """
    <style>
    /* Gradient background for a modern look */
    .stApp {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fad0c4);
    }
    /* Title styling */
    .title {
        text-align: center;
        color: #6a1b9a;
        font-size: 60px;
        font-weight: bold;
        text-shadow: 2px 2px #f8bbd0;
    }
    /* Subheader styling */
    .stMarkdown h2 {
        color: #1565c0;
        font-size: 30px;
        font-weight: 700;
        border-bottom: 2px solid #1565c0;
        padding-bottom: 5px;
    }
    /* Smaller headings */
    .stMarkdown h3 {
        color: #2e7d32;
        font-size: 22px;
        font-weight: 600;
    }
    /* General text */
    .stMarkdown, .stText {
        color: #4e342e;
    }
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(to right, #ff512f, #dd2476);
        color: white;
        border: none;
        font-weight: bold;
        border-radius: 10px;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🚗 Car Price Prediction App</h1>', unsafe_allow_html=True)

df = pd.read_csv("car data.csv")
model = pickle.load(open("model.pkl", "rb"))

st.subheader("Enter Car Details")
year = st.number_input("Enter Year", min_value=2000, max_value=2025, value=2019)
present_price = st.number_input("Enter Present Price (in lakhs)", value=5.0)
kms = st.number_input("Enter Kilometers Driven", value=30000)

if st.button("Predict Price"):
    prediction = model.predict([[year, present_price, kms]])
    
    # 🔥 Stylish Result Box (replaces green success box)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ff512f, #dd2476);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        margin-top: 10px;
    ">
        💰 Estimated Price: {prediction[0]:.2f} lakhs
    </div>
    """, unsafe_allow_html=True)

st.subheader("📊 Data Visualization")

st.write("### Year vs Selling Price")
plt.figure()
sns.scatterplot(x=df["Year"], y=df["Selling_Price"], color="#6a1b9a")
st.pyplot(plt)

st.write("### Kilometers vs Selling Price")
plt.figure()
sns.scatterplot(x=df["Kms_Driven"], y=df["Selling_Price"], color="#1565c0")
st.pyplot(plt)

st.write("### Selling Price Distribution")
plt.figure()
sns.histplot(df["Selling_Price"], bins=20, kde=True, color="#2e7d32")
st.pyplot(plt)