import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model= joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter the Fake/Real news below")

news_input=st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input=vectorizer.transform([news_input])
        prediction=model.predict(transform_input)

        if prediction[0]==1:
            st.success("Real news")
        else:
            st.error("Fake News")
    else:
        st.warning("Enter news first")
