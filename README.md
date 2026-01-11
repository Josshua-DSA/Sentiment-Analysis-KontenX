# Sentiment Analysis Konten di platform X

Proyek ini bertujuan untuk mengumpulkan data tweet dari Twitter menggunakan tool `tweet-harvest` dan menyimpan hasil scraping data menggunakan Python pandas. Setelah melakukan penyimpanan menjadi file csv, data akan dianalisis sentimennya menggunakan modul Sastrawi, nltk, dan beberapa modul lain.

## Deskripsi Proyek

Project ini melakukan scraping/crawling data tweet berdasarkan keyword tertentu dengan menggunakan Node.js package `tweet-harvest`. Data yang dikumpulkan kemudian disimpan dalam format CSV dan dianalisis menggunakan pandas.

## Prerequisites

### Software yang Dibutuhkan
- Node.js (versi 20.x)
- Python dengan library pandas,
- Sistem operasi berbasis Linux/Ubuntu (atau environment yang mendukung apt-get)

### Token Autentikasi
Diperlukan Twitter authentication token untuk mengakses API Twitter. Token ini harus ditempatkan pada variabel `twitter_auth_token`.

```python
twitter_auth_token = "your-twitter-auth-token-here"
```

## Instalasi

### 1. Update Sistem dan Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
```

### 2. Install Node.js

```bash
# Buat direktori untuk keyrings
sudo mkdir -p /etc/apt/keyrings

# Download dan install GPG key
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

# Tambahkan repository Node.js
NODE_MAJOR=20 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list

# Update dan install Node.js
sudo apt-get update
sudo apt-get install nodejs -y

# Verifikasi instalasi
node -v
```

## Cara Penggunaan

### 1. Konfigurasi Parameter Scraping

Tentukan parameter untuk scraping data:

```python
filename = 'BBM(kompas_7oct).csv'  # Nama file output
search_keyword = 'bensin since:2025-10-7 until:2025-10-14 lang:id'  # Query pencarian
limit = 300  # Jumlah maksimal tweet yang akan diambil
```

**Penjelasan Parameter Query:**
- `bensin` - keyword yang dicari
- `since:2025-10-7` - tanggal mulai pencarian
- `until:2025-10-14` - tanggal akhir pencarian
- `lang:id` - bahasa Indonesia

### 2. Jalankan Scraping

```bash
npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {limit} --token {twitter_auth_token}
```

Parameter:
- `-o` : Output filename
- `-s` : Search keyword/query
- `--tab` : Tab pencarian (LATEST untuk tweet terbaru)
- `-l` : Limit jumlah tweet
- `--token` : Authentication token

### 3. Load dan Analisis Data dengan Pandas

```python
import pandas as pd

# Tentukan path file CSV
file_path = f"tweets-data/{filename}"

# Baca file CSV
df = pd.read_csv(file_path, delimiter=",")

# Tampilkan DataFrame
display(df)
```

# Analisis Sentimen Tweet BBM

Dokumen ini menjelaskan proses lengkap analisis sentimen dari data tweet tentang BBM (Bahan Bakar Minyak) menggunakan Python.

---

## 1. Ekstraksi Dataset

Tahap awal adalah memuat dataset tweet yang telah di-scrape sebelumnya.

```python
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data CSV
csv_path = "bbm.csv"
df = pd.read_csv(csv_path)

# Tampilkan info dataset
print(f"Jumlah tweet: {len(df)}")
print(f"Kolom yang tersedia: {df.columns.tolist()}")
df.head()
```

**Output yang diharapkan:**
- Jumlah tweet yang berhasil di-load
- Kolom-kolom yang ada dalam dataset
- Preview 5 baris pertama data

---

## 2. Text Cleaning

Tahap pembersihan teks sangat penting untuk meningkatkan kualitas analisis sentimen.

### 2.1 Install Dependencies

```python
!pip install Sastrawi
```

### 2.2 Import Libraries dan Setup

```python
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Download NLTK data
nltk.download('punkt_tab')
```

### 2.3 Persiapan Stopwords dan Stemmer

```python
# Siapkan stopwords dari Sastrawi
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# Tambahkan stopword khas media sosial
extra_stopwords = {
    'yg', 'aja', 'nih', 'sih', 'dong', 'ya', 'aku', 'akuuu',
    'bang', 'banget', 'loh', 'lah', 'nya', 'deh', 'kak', 'bro',
    'dr', 'udh', 'udah', 'dah', 'gitu', 'gak', 'ga', 'gk'
}
stopwords.update(extra_stopwords)

# Siapkan stemmer
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()
```

### 2.4 Kamus Normalisasi

```python
# Kamus normalisasi kata tidak baku → baku
normalisasi = {
    'gk': 'tidak', 'ga': 'tidak', 'bgt': 'banget',
    'anjir': 'anjing', 'anjay': 'anjing', 'yg': 'yang', 
    'tp': 'tapi', 'aja': 'saja', 'klo': 'kalau', 
    'tdk': 'tidak', 'udh': 'sudah', 'dr': 'dari', 
    'dl': 'dulu', 'dgn': 'dengan', 'jg': 'juga', 
    'sm': 'sama', 'trs': 'terus', 'jgk': 'juga', 
    'trnyta': 'ternyata', 'bkn': 'bukan'
}
```

### 2.5 Fungsi Cleaning Utama

```python
def clean_tweet(text):
    """
    Fungsi untuk membersihkan teks tweet
    """
    # Pastikan input berupa string
    if not isinstance(text, str):
        return ""
    
    # 1. Hapus mention, hashtag, dan URL
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # 2. Hapus emoji dan karakter non-teks
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emotikon
        u"\U0001F300-\U0001F5FF"  # simbol & pictograf
        u"\U0001F680-\U0001F6FF"  # transportasi
        u"\U0001F1E0-\U0001F1FF"  # bendera
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # 3. Ubah ke huruf kecil
    text = text.lower()
    
    # 4. Hapus tanda baca, angka, dan karakter ganda
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # "bensinnnn" → "bensin"
    
    # 5. Tokenisasi
    tokens = word_tokenize(text)
    
    # 6. Normalisasi kata tidak baku
    tokens = [normalisasi.get(word, word) for word in tokens]
    
    # 7. Hilangkan stopwords
    tokens = [word for word in tokens if word not in stopwords and len(word) > 2]
    
    # 8. Stemming
    text_stemmed = stemmer.stem(' '.join(tokens))
    
    # 9. Hapus spasi berlebih
    text_stemmed = re.sub(r'\s+', ' ', text_stemmed).strip()
    
    return text_stemmed

# Terapkan fungsi cleaning
df['clean_text'] = df['full_text'].apply(clean_tweet)

# Lihat hasilnya
print(df[['full_text', 'clean_text']].head(10))
```

### 2.6 Deep Cleaning (Post-processing)

```python
# Kamus tambahan untuk perbaikan
kamus_perbaikan = {
    'malem': 'malam',
    'ujung': 'akhir',
    'bgt': 'banget',
    'emg': 'memang',
    'cmn': 'cuma',
    'ywdah': 'ya sudah',
    'ywdh': 'ya sudah',
    'aja': 'saja',
    'blm': 'belum',
    'ampun': 'kesal',
    'bener': 'benar',
    'lumayang': 'lumayan',
    'der': 'teman',
    'yaampun': 'ya ampun',
    'tlong': 'tolong',
    'bikin': 'membuat',
    'udh': 'sudah',
    'dr': 'dari',
    'tdk': 'tidak',
    'ga': 'tidak',
    'gk': 'tidak',
    'anjir': 'anjing',
    'tp': 'tapi'
}

# Stopword tambahan
extra_stopwords2 = {
    'udah', 'dah', 'aja', 'sih', 'deh', 'nih', 'dong',
    'loh', 'lah', 'bang', 'bro', 'kak', 'ko', 'mah', 'nya'
}
stopwords.update(extra_stopwords2)

def deep_clean_text(text):
    """
    Tahap lanjutan pembersihan teks
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Normalisasi tambahan
    words = text.split()
    words = [kamus_perbaikan.get(w, w) for w in words]
    
    # 2. Hapus kata duplikat berturut-turut
    cleaned_words = []
    for i, w in enumerate(words):
        if i == 0 or w != words[i-1]:
            cleaned_words.append(w)
    
    # 3. Hapus stopwords tambahan
    cleaned_words = [w for w in cleaned_words if w not in stopwords]
    
    # 4. Hapus token terlalu pendek (<3 huruf)
    cleaned_words = [w for w in cleaned_words if len(w) > 2]
    
    # 5. Gabungkan kembali
    text_final = ' '.join(cleaned_words)
    text_final = re.sub(r'\s+', ' ', text_final).strip()
    
    return text_final

# Terapkan deep cleaning
df['final_clean_text'] = df['clean_text'].apply(deep_clean_text)

# Lihat hasilnya
print(df[['clean_text', 'final_clean_text']].head(10))

# Hapus kolom intermediate dan rename
df.drop(columns=['clean_text'], inplace=True)
df.rename(columns={'final_clean_text': 'clean_text'}, inplace=True)

print("\n✅ Text cleaning selesai!")
df.head()
```

---

## 3. Exploratory Data Analysis (EDA)

Analisis eksploratori untuk memahami karakteristik data sebelum melakukan analisis sentimen.

### 3.1 Statistik Dasar

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Hitung jumlah kata per tweet
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# Statistik dasar
print("=" * 50)
print("STATISTIK DATASET")
print("=" * 50)
print(f"Jumlah Tweet      : {len(df)}")
print(f"Rata-rata panjang : {df['word_count'].mean():.2f} kata")
print(f"Tweet terpanjang  : {df['word_count'].max()} kata")
print(f"Tweet terpendek   : {df['word_count'].min()} kata")
print("=" * 50)
```

### 3.2 Frekuensi Kata

```python
# Gabungkan semua kata
all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words)

# Tampilkan 20 kata paling sering muncul
print("\n20 Kata Paling Sering Muncul:")
print("=" * 50)
for word, count in word_freq.most_common(20):
    print(f"{word:20s} : {count:4d}")
print("=" * 50)
```

### 3.3 Visualisasi Word Cloud

```python
# Generate Word Cloud
plt.figure(figsize=(12, 6))
wc = WordCloud(
    width=1200, 
    height=600, 
    background_color='white',
    collocations=False,
    colormap='viridis'
).generate(' '.join(df['clean_text']))

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud dari Tweet tentang BBM', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 3.4 Top 10 Kata (Bar Chart)

```python
# Visualisasi Top 10 kata
top_words = dict(word_freq.most_common(10))

plt.figure(figsize=(10, 6))
plt.barh(list(top_words.keys()), list(top_words.values()), color='skyblue', edgecolor='navy')
plt.gca().invert_yaxis()
plt.title("10 Kata Paling Sering Muncul", fontsize=14, fontweight='bold')
plt.xlabel("Frekuensi", fontsize=12)
plt.ylabel("Kata", fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 3.5 EDA dengan Custom Stopwords

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Setup stopwords
stop_words = set(stopwords.words('indonesian'))
custom_stopwords = {
    "aja", "ya", "nih", "dong", "kan", "lah", "sih",
    "kalo", "kalau", "gue", "gua", "aku", "kamu",
    "jadi", "sama", "apa", "yang", "itu", "di", "ke", "pada",
    "udah", "lagi", "buat", "dulu", "bukan", "cuma", "pas"
}
stop_words.update(custom_stopwords)

# Tokenisasi ulang dengan filtering
all_tokens = []
for text in df['clean_text']:
    tokens = word_tokenize(text)
    filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    all_tokens.extend(filtered_tokens)

# Hitung frekuensi
word_freq_clean = Counter(all_tokens)

# Tampilkan top 10
print("\n10 Kata Paling Sering Muncul (Setelah Filtering):")
print("=" * 50)
top_10 = word_freq_clean.most_common(10)
for w, c in top_10:
    print(f"{w:20s} : {c:4d}")
print("=" * 50)

# Word Cloud bersih
wordcloud_clean = WordCloud(
    width=1200, 
    height=600, 
    background_color='white',
    colormap='plasma'
).generate_from_frequencies(word_freq_clean)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_clean, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Bersih dari Tweet tentang BBM", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Bar chart top 10
plt.figure(figsize=(10, 6))
words = [w for w, _ in top_10[::-1]]
counts = [c for _, c in top_10[::-1]]
plt.barh(words, counts, color='coral', edgecolor='darkred')
plt.title("10 Kata Paling Sering Muncul (Cleaned)", fontsize=14, fontweight='bold')
plt.xlabel("Frekuensi", fontsize=12)
plt.ylabel("Kata", fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. Labelling dengan Lexicon-Based

Metode labelling otomatis menggunakan kamus kata positif dan negatif.

### 4.1 Definisi Lexicon

```python
# Lexicon kata positif
positive_words = [
    "bagus", "baik", "hebat", "luar biasa", "menyenangkan", "senang", "gembira",
    "puas", "ramah", "cepat", "terbaik", "suka", "mantap", "keren", "wow",
    "indah", "bersih", "nyaman", "positif", "memuaskan", "top", "rekomendasi",
    "love", "terimakasih", "makasih", "enak", "super", "juara", "segar",
    "menakjubkan", "bagus banget", "bagus sekali", "helpful", "mantul"
]

# Lexicon kata negatif
negative_words = [
    "buruk", "jelek", "parah", "kecewa", "lambat", "benci", "tidak puas",
    "mengecewakan", "sakit", "nggak enak", "susah", "error", "lemot",
    "gagal", "macet", "bete", "payah", "bohong", "ngecewain", "ga jelas",
    "sampah", "malas", "ngeselin", "ngecewa", "ngga suka", "gak suka",
    "berantakan", "rusak", "tidak baik", "tidak bagus", "menyedihkan",
    "ngecewain banget", "ribet", "pelayanan buruk", "parah banget"
]
```

### 4.2 Fungsi Scoring

```python
def get_sentiment_score(text):
    """
    Hitung skor sentimen berdasarkan lexicon
    
    Returns:
        int: Skor positif jika > 0, negatif jika < 0, netral jika = 0
    """
    words = re.findall(r'\w+', text.lower())
    score = 0
    
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    
    return score

def get_sentiment_label(score):
    """
    Konversi skor menjadi label sentimen
    """
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    else:
        return "netral"
```

### 4.3 Labelling Dataset

```python
# Hitung skor dan label
df["sentiment_score"] = df["clean_text"].apply(get_sentiment_score)
df["sentiment_lexicon"] = df["sentiment_score"].apply(get_sentiment_label)

# Tampilkan sample hasil
print("\nContoh Hasil Labelling:")
print("=" * 80)
print(df[["clean_text", "sentiment_score", "sentiment_lexicon"]].head(10))

# Distribusi label
print("\n" + "=" * 80)
print("DISTRIBUSI LABEL SENTIMEN")
print("=" * 80)
sentiment_dist = df["sentiment_lexicon"].value_counts()
print(sentiment_dist)
print("=" * 80)

# Visualisasi distribusi
plt.figure(figsize=(8, 6))
sentiment_dist.plot(kind='bar', color=['green', 'gray', 'red'], edgecolor='black')
plt.title('Distribusi Label Sentimen', fontsize=14, fontweight='bold')
plt.xlabel('Label Sentimen', fontsize=12)
plt.ylabel('Jumlah Tweet', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 4.4 Simpan Hasil ke CSV

```python
# Pilih kolom yang relevan
keep_cols = [c for c in [
    "created_at", "full_text", "text", "clean_text",
    "sentiment_score", "sentiment_lexicon"
] if c in df.columns]

# Simpan ke file
df[keep_cols].to_csv("lexicon.csv", index=False, encoding="utf-8-sig")
print("\n✅ Dataset berlabel tersimpan ke: lexicon.csv")

# Preview hasil
df[keep_cols].head()
```

---

## 5. Supervised Learning (KNN & Naive Bayes)

Menggunakan machine learning untuk klasifikasi sentimen dengan membandingkan dua algoritma.

### 5.1 Import Libraries

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
```

### 5.2 Preprocessing Data

```python
# Load dataset berlabel
df = pd.read_csv('lexicon.csv')

print(f"Total data: {len(df)}")
print(f"Distribusi label:\n{df['sentiment_lexicon'].value_counts()}")

# Encode label menjadi angka
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment_lexicon'])

print(f"\nMapping label:")
for i, label in enumerate(le.classes_):
    print(f"  {label} → {i}")
```

### 5.3 Split Data

```python
# Split train-test (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']  # menjaga proporsi label
)

print(f"\nJumlah data training: {len(X_train)}")
print(f"Jumlah data testing : {len(X_test)}")
```

### 5.4 Feature Extraction (TF-IDF)

```python
# Vectorize dengan TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nUkuran feature matrix: {X_train_tfidf.shape}")
print(f"Jumlah vocabulary: {len(tfidf.vocabulary_)}")
```

### 5.5 Training Model KNN

```python
print("\n" + "=" * 60)
print("TRAINING MODEL K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# Train KNN dengan k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

# Prediksi
knn_pred = knn.predict(X_test_tfidf)

# Evaluasi
knn_acc = accuracy_score(y_test, knn_pred)
print(f"\n✅ Akurasi KNN: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
```

### 5.6 Training Model Naive Bayes

```python
print("\n" + "=" * 60)
print("TRAINING MODEL NAIVE BAYES")
print("=" * 60)

# Train Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Prediksi
nb_pred = nb.predict(X_test_tfidf)

# Evaluasi
nb_acc = accuracy_score(y_test, nb_pred)
print(f"\n✅ Akurasi Naive Bayes: {nb_acc:.4f} ({nb_acc*100:.2f}%)")
```

### 5.7 Laporan Evaluasi Lengkap

```python
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT - NAIVE BAYES")
print("=" * 60)
print(classification_report(
    y_test, 
    nb_pred, 
    target_names=le.classes_,
    digits=4
))

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT - KNN")
print("=" * 60)
print(classification_report(
    y_test, 
    knn_pred, 
    target_names=le.classes_,
    digits=4
))
```

### 5.8 Confusion Matrix

```python
# Confusion Matrix - Naive Bayes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Naive Bayes
cm_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0])
axes[0].set_title(f'Confusion Matrix - Naive Bayes\nAccuracy: {nb_acc:.2%}', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# KNN
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1])
axes[1].set_title(f'Confusion Matrix - KNN\nAccuracy: {knn_acc:.2%}', 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
```

### 5.9 Perbandingan Model

```python
# Visualisasi perbandingan akurasi
models = ['Naive Bayes', 'KNN']
accuracies = [nb_acc, knn_acc]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'], 
               edgecolor='black', linewidth=2)

# Tambahkan nilai di atas bar
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Perbandingan Akurasi Model', fontsize=14, fontweight='bold')
plt.ylabel('Akurasi', fontsize=12)
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Kesimpulan
print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)
if nb_acc > knn_acc:
    print(f"✅ Naive Bayes lebih baik dengan selisih {(nb_acc - knn_acc)*100:.2f}%")
elif knn_acc > nb_acc:
    print(f"✅ KNN lebih baik dengan selisih {(knn_acc - nb_acc)*100:.2f}%")
else:
    print("⚖️ Kedua model memiliki performa yang sama")
print("=" * 60)
```

### 5.10 Prediksi Tweet Baru

```python
def predict_sentiment(text, model='nb'):
    """
    Fungsi untuk memprediksi sentimen dari teks baru
    
    Args:
        text (str): Teks yang akan diprediksi
        model (str): 'nb' untuk Naive Bayes, 'knn' untuk KNN
    
    Returns:
        str: Label sentimen prediksi
    """
    # Cleaning
    cleaned = deep_clean_text(clean_tweet(text))
    
    # Vectorize
    vectorized = tfidf.transform([cleaned])
    
    # Predict
    if model == 'nb':
        pred = nb.predict(vectorized)[0]
    else:
        pred = knn.predict(vectorized)[0]
    
    # Decode label
    sentiment = le.inverse_transform([pred])[0]
    
    return sentiment

# Test prediksi
test_tweets = [
    "Harga bensin naik lagi, parah banget!",
    "Syukurlah harga BBM stabil, bagus untuk ekonomi",
    "Bensin hari ini tersedia di SPBU"
]

print("\n" + "=" * 60)
print("UJI PREDIKSI TWEET BARU")
print("=" * 60)
for tweet in test_tweets:
    pred_nb = predict_sentiment(tweet, 'nb')
    pred_knn = predict_sentiment(tweet, 'knn')
    print(f"\nTweet: {tweet}")
    print(f"  → Naive Bayes: {pred_nb}")
    print(f"  → KNN        : {pred_knn}")
print("=" * 60)
```

---

## Rangkuman Proses

### Pipeline Lengkap:
1. **Ekstraksi Data** → Load dataset tweet dari CSV
2. **Text Cleaning** → Normalisasi, stemming, stopword removal
3. **EDA** → Analisis frekuensi kata, word cloud, statistik
4. **Labelling** → Lexicon-based sentiment scoring
5. **Machine Learning** → Training KNN & Naive Bayes
6. **Evaluasi** → Accuracy, precision, recall, F1-score
7. **Deployment** → Fungsi prediksi untuk tweet baru

### Hasil yang Diharapkan:
- Dataset bersih siap analisis
- Visualisasi insight data
- Model terlatih dengan akurasi tinggi
- Kemampuan prediksi sentimen otomatis

---

## Tips & Best Practices

1. **Text Cleaning**: Semakin bersih data, semakin baik hasil model
2. **Lexicon**: Perluas kamus kata positif/negatif untuk akurasi lebih baik
3. **Feature Engineering**: Coba TF-IDF dengan parameter berbeda
4. **Hyperparameter Tuning**: Eksperimen dengan nilai K pada KNN
5. **Cross-Validation**: Gunakan k-fold CV untuk evaluasi lebih robust

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
nltk
Sastrawi
scikit-learn
wordcloud
```

Install semua dengan:
```bash
pip install pandas numpy matplotlib seaborn nltk Sastrawi scikit-learn wordcloud
```

## Struktur Project

```
project/
│
├── tweets-data/           # Folder untuk menyimpan hasil scraping
│   └── BBM(kompas_7oct).csv
│
└── README.md             # Dokumentasi project
```

## Contoh Use Case

Project ini digunakan untuk mengumpulkan data tweet tentang topik "bensin" dalam rentang waktu 7-14 Oktober 2025, dengan filter bahasa Indonesia. Data ini dapat digunakan untuk:

- Analisis sentimen publik
- Monitoring trending topics
- Penelitian media sosial
- Analisis opini publik terhadap isu tertentu

## Catatan Penting

⚠️ **Keamanan Token**: Jangan pernah share atau commit authentication token ke repository publik.

⚠️ **Rate Limiting**: Twitter API memiliki batasan request. Pastikan untuk mematuhi rate limit yang berlaku.

⚠️ **Compliance**: Pastikan penggunaan data sesuai dengan Terms of Service Twitter dan peraturan privasi yang berlaku.

## Troubleshooting

### Node.js tidak terinstall
Pastikan semua perintah instalasi dijalankan dengan benar dan sistem sudah di-update.

### Error saat scraping
- Periksa kembali authentication token
- Pastikan format query sudah benar
- Cek koneksi internet

### File CSV tidak ditemukan
Pastikan path file sudah benar dan proses scraping berhasil dijalankan.

## Lisensi

Project ini menggunakan tool `tweet-harvest` yang memiliki lisensi tersendiri. Pastikan untuk mematuhi lisensi dan ketentuan penggunaan.

## Kontributor

[Joshua Remedial Syeba / Kelompok 4]
Credit : Helmi Satria
yt     : https://www.youtube.com/@helmisatria

---

**Disclaimer**: Project ini dibuat untuk tujuan edukasi dan penelitian. Pengguna bertanggung jawab penuh atas penggunaan data yang dikumpulkan.
