import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import tensorflow as tf


# Load the model and scaler
model = tf.keras.models.load_model("model.keras")

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Title and description
st.title("🎓 Graduate Admission Prediction")
st.markdown("""
Welcome to the Graduate Admission Prediction App!  
This app uses an Artificial Neural Network (ANN) to predict your chances of admission to graduate school based on various factors.  
Simply adjust the sliders below with your details and click **Predict** to see your estimated chance of admission.
""")

# Input features
st.header("📊 Input Your Details")

gre_score = st.slider("GRE Score", min_value=260, max_value=340, value=300, step=1)
toefl_score = st.slider("TOEFL Score", min_value=80, max_value=120, value=100, step=1)
university_rating = st.slider("University Rating", min_value=1, max_value=5, value=3, step=1)
sop = st.slider("Statement of Purpose (SOP) Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
lor = st.slider("Letter of Recommendation (LOR) Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
cgpa = st.slider("Undergraduate GPA (CGPA)", min_value=6.0, max_value=10.0, value=8.0, step=0.1)
research = st.selectbox("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("🔮 Predict Chance of Admission"):
    # Prepare input data
    input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]

    # Display result
    st.success(f"🎉 Your predicted chance of admission is **{prediction:.2%}**")

    # Interpretation
    if prediction >= 0.8:
        st.info("High chance! You're likely to get admitted.")
    elif prediction >= 0.6:
        st.info("Good chance! Keep improving your profile.")
    elif prediction >= 0.4:
        st.warning("Moderate chance. Consider strengthening weaker areas.")
    else:
        st.error("Low chance. Focus on improving your scores and experiences.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")
