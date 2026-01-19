# 📱 Classification Smartphone Performance

Proyek ini adalah aplikasi web berbasis Machine Learning yang dirancang untuk membantu pengguna mengklasifikasikan kelas harga dan potensi penggunaan terbaik dari sebuah smartphone berdasarkan spesifikasi teknisnya. Dikembangkan menggunakan **Streamlit** dan algoritma **Random Forest**, aplikasi ini mampu memberikan prediksi akurat serta rekomendasi perangkat serupa.

## 🌟 Fitur Utama

Aplikasi ini dibagi menjadi tiga bagian utama:

1. **🔮 Prediksi Pintar**: Masukkan spesifikasi teknis (RAM, Processor, Baterai, Kamera, dll.) untuk mendapatkan prediksi instan mengenai kelas harga dan kategori penggunaan (Gaming, Photography, dll.).
2. **📊 Visualisasi Dataset**: Eksplorasi data interaktif yang menampilkan distribusi harga per brand dan statistik fitur utama dari pasar smartphone saat ini.
3. **ℹ️ Info Model**: Transparansi mengenai algoritma yang digunakan, daftar fitur (features) yang berpengaruh, dan parameter model.
4. **💰 Estimasi Harga Multicurrency**: Menampilkan rentang harga dalam mata uang Rupee (India) dan Rupiah (Indonesia).
5. **📱 Rekomendasi Serupa**: Memberikan saran 5 smartphone dari dataset yang memiliki spesifikasi paling mirip dengan input pengguna.

## 🛠️ Teknologi & Library

Proyek ini dibangun menggunakan ekosistem Python modern:

* **Framework**: [Streamlit](https://streamlit.io/) (Interface aplikasi web)
* **Data Science**: Pandas & NumPy (Manipulasi data)
* **Machine Learning**: Scikit-learn (Algoritma & Pemrosesan), Imbalanced-learn
* **Visualisasi**: Plotly, Matplotlib, Seaborn (Grafik interaktif)
* **Model Deployment**: Joblib (Serialisasi model)

## 📋 Prasyarat

Sebelum menjalankan proyek ini secara lokal, pastikan Anda telah menginstal:

* Python 3.9 atau lebih baru.
* `pip` (Python package manager).

## 🚀 Instalasi & Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda:

1. **Clone Repository**
```bash
git clone https://github.com/mikaelazzz/classification-smartphone-performance.git
cd Classification-Smartphone-Performance-master

```


2. **Buat Virtual Environment** (Sangat disarankan)
```bash
python -m venv venv
# Aktivasi (Windows)
venv\Scripts\activate
# Aktivasi (Linux/Mac)
source venv/bin/activate

```


3. **Instal Dependensi**
```bash
pip install -r requirements.txt

```


4. **Persiapkan Model**
Pastikan file model `.pkl` tersedia di folder `model/`. Jika belum, jalankan notebook pelatihan terlebih dahulu:
```bash
jupyter notebook notebook/smartphone_performance_classification.ipynb

```


5. **Jalankan Aplikasi**
```bash
streamlit run view/app.py

```



## 📊 Informasi Dataset

Dataset yang digunakan mencakup ribuan entri data smartphone dari berbagai brand dengan fitur lengkap.

* **Sumber**: [Kaggle - Smartphones Dataset](https://www.kaggle.com/datasets/nishantdeswal1810/smartphones)
* **Target Klasifikasi**:
* **Harga**: Entry Level, Mid Range, High End, Flagship.
* **Penggunaan**: Gaming, Daily Use, Photography, Business, All-Rounder.



## ⚙️ Detail Machine Learning

Aplikasi menggunakan algoritma **Random Forest Classifier** yang dioptimalkan dengan konfigurasi berikut:

* **Estimators**: 200 pepohonan (trees).
* **Max Depth**: 8 (Untuk mencegah overfitting).
* **Class Weight**: Balanced (Menangani data yang tidak seimbang).

**Fitur yang Digunakan**: Kecepatan prosesor, jumlah core, RAM, memori internal, kapasitas baterai, ukuran layar, refresh rate, resolusi, jumlah kamera, dukungan 5G, NFC, dan pengisian cepat.
