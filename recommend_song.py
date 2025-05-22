import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset.csv")

# df'in track_name sütunundan rastgele 3 şarkı seçip liste olarak yazdır
liked_songs = df['track_name'].dropna().sample(3).tolist()

# Şarkı isimlerine göre filtrele
liked_df = df[df['track_name'].isin(liked_songs)]

# Kullanılacak özellikler (numerik sütunlar)
feature_cols = [
    'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# Null değerleri olan satırları at
df = df.dropna(subset=feature_cols)

# Özellikleri ölçeklendir (standardizasyon)
scaler = StandardScaler()
features = scaler.fit_transform(df[feature_cols])

# Beğenilen şarkıların ortalama vektörünü al
liked_features = df[df['track_name'].isin(liked_songs)]
liked_scaled = scaler.transform(liked_features[feature_cols])
user_profile = liked_scaled.mean(axis=0).reshape(1, -1)

# Tüm şarkılarla benzerliği hesapla
similarities = cosine_similarity(user_profile, features).flatten()

# Sonuçları DataFrame'e ekle
df['similarity'] = similarities

# Zaten beğenilenleri çıkar
recommendations = df[~df['track_name'].isin(liked_songs)]

# Benzerliğe göre sırala
top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(10)

# Önerilen şarkıları göster
print("Beğenilen şarkılar:")
print(*liked_songs, sep="\n")
print("-"*50)
print("Önerilen şarkılar:")
print(*top_recommendations['track_name'].unique(), sep="\n")
