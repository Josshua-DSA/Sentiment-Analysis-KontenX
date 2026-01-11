# Sentiment-Analysis-KontenX
# Twitter Data Scraping Project

Proyek ini bertujuan untuk mengumpulkan data tweet dari Twitter menggunakan tool `tweet-harvest` dan menganalisisnya menggunakan Python pandas.

## Deskripsi Proyek

Project ini melakukan scraping/crawling data tweet berdasarkan keyword tertentu dengan menggunakan Node.js package `tweet-harvest`. Data yang dikumpulkan kemudian disimpan dalam format CSV dan dianalisis menggunakan pandas.

## Prerequisites

### Software yang Dibutuhkan
- Node.js (versi 20.x)
- Python dengan library pandas
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
