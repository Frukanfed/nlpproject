import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# nltk paketleri indirme
nltk.download("stopwords")

# CSV dosyasını okuma
df = pd.read_csv("./data/data.csv")


# Metin işleme fonksiyonu
def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    # Newline ve tab karakterlerinin boşluk ile değiştirilmesi
    text = text.replace("\n", " ").replace("\t", " ")
    # Sayısal karakterlerin kaldırılması
    text = re.sub(r"\d+", "", text)
    # Noktalama işaretlerinin kaldırılması
    text = re.sub(r"[^\w\s]", "", text)
    # Durma kelimelerinin kaldırılması (ve, veya, ile vb.)
    stop_words = set(stopwords.words("turkish"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Stemming (pek işe yaramamakta)
    # stemmer = SnowballStemmer("turkish")
    # text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# Birinci kolondaki metinlere işlem uygulama
df["Haber Gövdesi"] = df["Haber Gövdesi"].apply(preprocess_text)

# İşlenmiş veriyi yeni bir CSV dosyasına kaydetme
df.to_csv("./data/processed_data.csv", index=False)

print("Veri başarıyla işlendi ve 'processed_data.csv' dosyasına kaydedildi.")
