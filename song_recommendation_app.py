import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa başlığı
st.title('Şarkı Öneri Sistemi')
st.write('Beğendiğiniz şarkıları seçin ve size benzer şarkılar önerelim!')

# Veri setini yükle
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    # Null değerleri olan satırları at (özellik sütunları için)
    feature_cols = [
        'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo'
    ]
    return df.dropna(subset=feature_cols)

# Veriyi yükle
df = load_data()

# Kullanılacak özellikler (numerik sütunlar)
feature_cols = [
    'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# Şarkı listesini hazırla
song_list = df['track_name'].dropna().unique().tolist()

# Çoklu seçim kutusu
st.subheader('Şarkı Seçimi')
selected_songs = st.multiselect(
    'Beğendiğiniz şarkıları seçin:',
    song_list
)

# Öneri butonu
if st.button('Şarkı Öner', key='recommend_button') and selected_songs:
    # Özellikleri ölçeklendir (standardizasyon)
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols])
    
    # Beğenilen şarkıların ortalama vektörünü al
    liked_features = df[df['track_name'].isin(selected_songs)]
    
    if not liked_features.empty:
        liked_scaled = scaler.transform(liked_features[feature_cols])
        user_profile = liked_scaled.mean(axis=0).reshape(1, -1)
        
        # Tüm şarkılarla benzerliği hesapla
        similarities = cosine_similarity(user_profile, features).flatten()
        
        # Sonuçları DataFrame'e ekle
        df_temp = df.copy()
        df_temp['similarity'] = similarities
        
        # Zaten beğenilenleri çıkar
        recommendations = df_temp[~df_temp['track_name'].isin(selected_songs)]
        
        # Benzersiz şarkı isimleri için track_id'ye göre gruplandır ve her gruptan en yüksek benzerlik skoruna sahip olanı seç
        # Her track_name için en yüksek benzerlik skoruna sahip satırı seç
        recommendations_unique = recommendations.loc[recommendations.groupby('track_name')['similarity'].idxmax()]
        
        # Benzerliğe göre sırala ve ilk 5 öneriyi al
        top_recommendations = recommendations_unique.sort_values(by='similarity', ascending=False).head(5)
        
        # Önerilen şarkıları göster
        st.subheader('Önerilen Şarkılar')
        for i, (_, row) in enumerate(top_recommendations.iterrows(), 1):
            st.write(f"{i}. {row['track_name']} - {row['artists']}")
        
        # Ek bilgiler
        st.write('\n**Öneri Detayları:**')
        st.write(f"Seçilen şarkı sayısı: {len(selected_songs)}")
        st.write(f"Önerilen şarkı sayısı: {len(top_recommendations)}")
    else:
        st.error('Seçilen şarkılar veri setinde bulunamadı. Lütfen başka şarkılar seçin.')
else:
    st.warning('Lütfen en az bir şarkı seçin.')

# Yan panel bilgileri
with st.sidebar:
    st.header('Nasıl Çalışır?')
    st.write("""
    1. Listeden beğendiğiniz şarkıları seçin
    2. 'Şarkı Öner' butonuna tıklayın
    3. Algoritma, seçtiğiniz şarkılara benzer 5 şarkı önerecektir
    
    Bu öneri sistemi, şarkıların çeşitli özelliklerini (danceability, energy, tempo vb.) 
    analiz ederek benzerlik hesaplar ve size en uygun şarkıları önerir.
    """)
    
    st.header('Veri Seti Hakkında')
    st.write(f"Toplam şarkı sayısı: {len(df)}")
    st.write(f"Benzersiz şarkı sayısı: {len(song_list)}")