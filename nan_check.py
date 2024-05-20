import pandas as pd

# CSV dosyasını yükleyin
df = pd.read_csv("processed_data.csv")

# NaN verileri kontrol edin
nan_var_mi = df.isnull().values.any()
print(f"NaN verisi var mı: {nan_var_mi}")

# NaN verilerin sayısını sütun bazında kontrol edin
print("Her sütundaki NaN verilerin sayısı:")
print(df.isnull().sum())

# NaN satırlarını silin
df_temiz = df.dropna()

# Temizlenmiş veri setini yeniden kaydedin
df_temiz.to_csv("processed_nachecked_data.csv", index=False)

print("NaN veriler temizlendi ve dosya yeniden kaydedildi.")
