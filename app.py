```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #ede9fe 100%);
}

/* Header */
.header {
    padding: 35px 25px;
    border-radius: 22px;
    text-align: center;
    color: white;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.30);
    margin-bottom: 30px;
}

.header h1 {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.header p {
    font-size: 18px;
    margin-top: 8px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.95);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.10);
    border: 1px solid rgba(255,255,255,0.8);
    margin-bottom: 20px;
}

/* Section titles */
.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #312e81;
    margin-bottom: 15px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    color: white;
    font-size: 19px;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 7px 18px rgba(79,70,229,0.35);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(79,70,229,0.45);
}

/* Result card */
.result-card {
    padding: 35px;
    margin-top: 25px;
    text-align: center;
    border-radius: 22px;
    background: white;
    box-shadow: 0 10px 35px rgba(0,0,0,0.15);
    animation: resultAnimation 0.8s ease;
}

@keyframes resultAnimation {
    0% {
        opacity: 0;
        transform: scale(0.85) translateY(20px);
    }
    60% {
        transform: scale(1.04);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

/* Prediction */
.prediction {
    font-size: 48px;
    font-weight: 850;
    color: #4f46e5;
    margin: 10px;
}

/* Confidence */
.confidence {
    font-size: 18px;
    font-weight: 600;
    color: #475569;
}

/* Info box */
.info-box {
    padding: 15px;
    border-radius: 12px;
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "GradientBoosting(1).pkl"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found. Please keep "
        "'GradientBoosting(1).pkl' in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()

except Exception as e:
    st.error("❌ Unable to load the Gradient Boosting model.")
    st.code(str(e))
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <h1>🤖 Gradient Boosting Predictor</h1>

    <p>
        Intelligent Machine Learning Classification Dashboard
    </p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Information")

    st.markdown("""
    **Algorithm**

    Gradient Boosting Classifier

    **Input Features**

    • Age  
    • Gender  
    • Review  
    • Education

    **Output**

    • No  
    • Yes
    """)

    st.markdown("---")

    st.markdown("""
    <div class="info-box">

    💡 <b>Tip</b><br>
    Select the categorical values and enter
    the age before clicking Predict.

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Enter Prediction Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# ============================================================
# LEFT CARD
# ============================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 👤 Personal Information")

    age = st.number_input(
        "Age",
```
