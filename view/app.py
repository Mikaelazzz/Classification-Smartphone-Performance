import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kinerja Smartphone",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load model dan scaler
@st.cache_resource
def load_model():
    """Load model Random Forest dan scaler"""
    try:
        # Coba load dengan joblib terlebih dahulu (lebih stabil)
        model_path_joblib = Path('model/smartphone_performance_rf_model.pkl')
        scaler_path = Path('model/smartphone_performance_scaler.pkl')
        features_path = Path('model/smartphone_performance_features.pkl')
        
        # Load model
        try:
            model = joblib.load(model_path_joblib)
        except:
            # Fallback ke pickle dengan encoding
            with open(model_path_joblib, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
        
        # Load scaler
        try:
            scaler = joblib.load(scaler_path)
        except:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f, encoding='latin1')
        
        # Load features
        try:
            features = joblib.load(features_path)
        except:
            with open(features_path, 'rb') as f:
                features = pickle.load(f, encoding='latin1')
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("💡 Tip: Pastikan model dilatih dengan Python versi yang kompatibel")
        return None, None, None

# Load data untuk visualisasi
@st.cache_data
def load_data():
    """Load dataset untuk visualisasi"""
    try:
        df = pd.read_csv('dataset/train.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Fungsi untuk mapping price range ke kategori
def get_category_name(price_range):
    """Mapping price range ke nama kategori"""
    categories = {
        0: "Entry Level (Harian)",
        1: "Mid Range",
        2: "High End",
        3: "Flagship (Gaming)"
    }
    return categories.get(price_range, "Unknown")

# Fungsi untuk mendapatkan warna berdasarkan kategori
def get_category_color(price_range):
    """Mendapatkan warna untuk setiap kategori"""
    colors = {
        0: "#ff6b6b",  # Red
        1: "#feca57",  # Yellow
        2: "#48dbfb",  # Blue
        3: "#1dd1a1"   # Green
    }
    return colors.get(price_range, "#95a5a6")

# Header
st.markdown('<p class="main-header">📱 Klasifikasi Kinerja Prosesor Smartphone</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/smartphone.png", width=100)
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini membantu mengklasifikasikan smartphone berdasarkan kinerja prosesornya:
    
    - **Entry Level**: HP untuk penggunaan harian
    - **Mid Range**: HP dengan performa menengah
    - **High End**: HP dengan performa tinggi
    - **Flagship**: HP kelas gaming
    
    **Algoritma**: Random Forest Classifier
    """)
    
    st.markdown("---")
    st.markdown("### Fitur Utama")
    st.markdown("""
    - Jumlah Core Prosesor
    - Kecepatan Clock (GHz)
    - RAM (MB)
    """)

# Load model
model, scaler, features = load_model()
df = load_data()

if model is None or scaler is None:
    st.error("⚠️ Gagal memuat model. Pastikan file model tersedia di folder 'models/'")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Visualisasi Data", "ℹ️ Informasi Model"])

# Tab 1: Prediksi
with tab1:
    st.markdown('<p class="sub-header">Masukkan Spesifikasi Smartphone</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_cores = st.slider(
            "Jumlah Core Prosesor",
            min_value=1,
            max_value=8,
            value=4,
            help="Jumlah core pada prosesor smartphone"
        )
    
    with col2:
        clock_speed = st.slider(
            "Kecepatan Clock (GHz)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Kecepatan clock prosesor dalam GHz"
        )
    
    with col3:
        ram = st.slider(
            "RAM (MB)",
            min_value=256,
            max_value=4000,
            value=2000,
            step=128,
            help="Kapasitas RAM dalam MB"
        )
    
    # Tombol prediksi
    if st.button("🔍 Prediksi Kelas Performa", type="primary", use_container_width=True):
        # Prepare input data dengan SEMUA fitur yang dibutuhkan model
        # Model dilatih dengan 10 fitur: battery_power, clock_speed, n_cores, ram, 
        # int_memory, mobile_wt, pc, fc, px_height, px_width
        
        # Nilai default untuk fitur tambahan (median dari dataset)
        input_data = pd.DataFrame({
            'battery_power': [1200],      # Default: median battery power
            'clock_speed': [clock_speed],  # User input
            'n_cores': [n_cores],          # User input
            'ram': [ram],                  # User input
            'int_memory': [32],            # Default: median internal memory
            'mobile_wt': [140],            # Default: median mobile weight
            'pc': [8],                     # Default: median primary camera
            'fc': [4],                     # Default: median front camera
            'px_height': [600],            # Default: median pixel height
            'px_width': [1200]             # Default: median pixel width
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Tampilkan hasil
        st.markdown("---")
        st.markdown('<p class="sub-header">Hasil Prediksi</p>', unsafe_allow_html=True)
        
        category_name = get_category_name(prediction)
        category_color = get_category_color(prediction)
        
        # Box hasil prediksi
        st.markdown(f"""
            <div style="background-color: {category_color}; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">Kelas: {category_name}</h2>
                <p style="color: white; font-size: 1.2rem; margin-top: 0.5rem;">
                    Confidence: {prediction_proba[prediction]*100:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Probabilitas untuk semua kelas
        st.markdown("### Probabilitas untuk Semua Kelas")
        
        prob_df = pd.DataFrame({
            'Kelas': [get_category_name(i) for i in range(4)],
            'Probabilitas': prediction_proba * 100
        })
        
        fig = px.bar(
            prob_df,
            x='Kelas',
            y='Probabilitas',
            color='Probabilitas',
            color_continuous_scale='Viridis',
            text='Probabilitas'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Probabilitas (%)",
            xaxis_title="Kelas Performa"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi
        st.markdown("### 💡 Interpretasi")
        if prediction == 0:
            st.info("📱 Smartphone ini cocok untuk penggunaan harian seperti browsing, social media, dan aplikasi ringan.")
        elif prediction == 1:
            st.info("📱 Smartphone ini memiliki performa menengah, cocok untuk multitasking dan gaming ringan.")
        elif prediction == 2:
            st.info("📱 Smartphone ini memiliki performa tinggi, cocok untuk gaming menengah dan aplikasi berat.")
        else:
            st.success("🎮 Smartphone ini adalah kelas flagship, sangat cocok untuk gaming berat dan aplikasi profesional!")

# Tab 2: Visualisasi Data
with tab2:
    if df is not None:
        st.markdown('<p class="sub-header">Distribusi Data Training</p>', unsafe_allow_html=True)
        
        # Distribusi Price Range
        col1, col2 = st.columns(2)
        
        with col1:
            price_counts = df['price_range'].value_counts().sort_index()
            fig1 = px.pie(
                values=price_counts.values,
                names=[get_category_name(i) for i in price_counts.index],
                title="Distribusi Kelas Performa",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Statistik deskriptif
            st.markdown("#### Statistik Fitur Utama")
            stats_df = df[['n_cores', 'clock_speed', 'ram']].describe()
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        # Scatter plot 3D
        st.markdown("#### Visualisasi 3D: Core vs Clock Speed vs RAM")
        fig_3d = px.scatter_3d(
            df,
            x='n_cores',
            y='clock_speed',
            z='ram',
            color='price_range',
            color_continuous_scale='Viridis',
            labels={
                'n_cores': 'Jumlah Core',
                'clock_speed': 'Clock Speed (GHz)',
                'ram': 'RAM (MB)',
                'price_range': 'Kelas'
            },
            title="Distribusi 3D Fitur Utama"
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Box plots
        st.markdown("#### Distribusi Fitur per Kelas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_box1 = px.box(
                df,
                x='price_range',
                y='n_cores',
                color='price_range',
                title="Jumlah Core per Kelas",
                labels={'price_range': 'Kelas', 'n_cores': 'Jumlah Core'}
            )
            st.plotly_chart(fig_box1, use_container_width=True)
        
        with col2:
            fig_box2 = px.box(
                df,
                x='price_range',
                y='clock_speed',
                color='price_range',
                title="Clock Speed per Kelas",
                labels={'price_range': 'Kelas', 'clock_speed': 'Clock Speed (GHz)'}
            )
            st.plotly_chart(fig_box2, use_container_width=True)
        
        with col3:
            fig_box3 = px.box(
                df,
                x='price_range',
                y='ram',
                color='price_range',
                title="RAM per Kelas",
                labels={'price_range': 'Kelas', 'ram': 'RAM (MB)'}
            )
            st.plotly_chart(fig_box3, use_container_width=True)

# Tab 3: Informasi Model
with tab3:
    st.markdown('<p class="sub-header">Informasi Model</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Tujuan Model")
        st.markdown("""
        Model ini dirancang untuk membantu orang awam memahami kelas performa smartphone berdasarkan spesifikasi prosesor:
        
        - **Entry Level (0)**: HP untuk penggunaan harian
        - **Mid Range (1)**: HP dengan performa menengah
        - **High End (2)**: HP dengan performa tinggi
        - **Flagship (3)**: HP kelas gaming
        """)
        
        st.markdown("### 🔧 Algoritma")
        st.markdown("""
        **Random Forest Classifier**
        
        Random Forest adalah ensemble learning method yang menggunakan multiple decision trees untuk membuat prediksi yang lebih akurat dan stabil.
        
        **Keunggulan:**
        - Akurasi tinggi
        - Robust terhadap overfitting
        - Dapat menangani data non-linear
        """)
    
    with col2:
        st.markdown("### 📊 Fitur Input")
        st.markdown("""
        Model menggunakan 10 fitur, dengan 3 fitur utama yang dapat diatur:
        
        **Fitur Utama (User Input):**
        1. **Jumlah Core Prosesor (n_cores)**
           - Range: 1-8 cores
           - Semakin banyak core, semakin baik multitasking
        
        2. **Kecepatan Clock (clock_speed)**
           - Range: 0.5-3.0 GHz
           - Menentukan kecepatan pemrosesan
        
        3. **RAM**
           - Range: 256-4000 MB
           - Mempengaruhi kemampuan menjalankan aplikasi
        
        **Fitur Tambahan (Nilai Default):**
        - Battery Power, Internal Memory, Mobile Weight
        - Primary Camera, Front Camera
        - Pixel Resolution (Height & Width)
        """)
        
        st.markdown("### 📈 Dataset")
        st.markdown("""
        - **Sumber**: Mobile Price Classification Dataset
        - **Jumlah Data**: 2000 samples
        - **Distribusi**: Balanced (25% per kelas)
        - **Total Fitur**: 10 fitur hardware
        """)
    
    st.markdown("---")
    st.markdown("### 🎓 Cara Menggunakan")
    st.markdown("""
    1. Masukkan spesifikasi smartphone pada tab **Prediksi**
    2. Klik tombol **Prediksi Kelas Performa**
    3. Lihat hasil prediksi dan interpretasinya
    4. Eksplorasi visualisasi data pada tab **Visualisasi Data**
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>📱 Klasifikasi Kinerja Smartphone | Powered by Random Forest</p>
        <p>Dataset: Mobile Price Classification</p>
    </div>
""", unsafe_allow_html=True)
