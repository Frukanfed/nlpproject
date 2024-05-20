import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Modeli ve vektörleştiriciyi yükleyin
with open("logistic_regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit uygulaması
st.title("Haber Sınıflandırma Uygulaması")

user_input = st.text_area("Bir haber metni girin:")

if st.button("Tahmin Et"):
    if user_input:
        # Kullanıcı girdisini TF-IDF matrisine dönüştürün
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Model ile tahmin yapın
        prediction = classifier.predict(user_input_tfidf)
        st.write(f"Tahmin edilen sınıf: {prediction[0]}")
    else:
        st.write("Lütfen bir haber metni girin.")
