import streamlit as st
import joblib

model = joblib.load("regression.joblib")

size = st.number_input("size")
st.write("The current size is ", size)

bedrooms = st.number_input("number of bedrooms")
st.write("The current number of bedrooms is ", bedrooms)

garden = st.number_input("whether a house has a garden")
st.write("The current whether a house has a garden is ", garden)

pred = model.predict([[size, bedrooms, garden]])
st.write("pred is ", pred)
