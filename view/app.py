import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Klasifikasi Smartphone",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .entry-level { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .mid-range { background: linear-gradient(135deg, #f39c12, #d68910); }
    .high-end { background: linear-gradient(135deg, #3498db, #2980b9); }
    .flagship { background: linear-gradient(135deg, #9b59b6, #8e44ad); }
    .gaming { background: linear-gradient(135deg, #e74c3c, #9b59b6); }
    .daily-use { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .photography { background: linear-gradient(135deg, #e67e22, #f39c12); }
    .business { background: linear-gradient(135deg, #34495e, #2c3e50); }
    .all-rounder { background: linear-gradient(135deg, #1abc9c, #16a085); }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_models():
    """Load all models and artifacts"""
    try:
        model_dir = Path(__file__).parent.parent / 'model'
        
        price_model = joblib.load(model_dir / 'smartphone_price_model.pkl')
        usage_model = joblib.load(model_dir / 'smartphone_usage_model.pkl')
        scaler = joblib.load(model_dir / 'scaler.pkl')
        features = joblib.load(model_dir / 'features.pkl')
        usage_encoder = joblib.load(model_dir / 'usage_encoder.pkl')
        price_class_names = joblib.load(model_dir / 'price_class_names.pkl')
        
        return price_model, usage_model, scaler, features, usage_encoder, price_class_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

@st.cache_data
def load_data():
    """Load dataset for visualization"""
    try:
        data_path = Path(__file__).parent.parent / 'dataset' / 'smartphones.csv'
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load resources
price_model, usage_model, scaler, features, usage_encoder, price_class_names = load_models()
df = load_data()

# Header
st.markdown('<h1 class="main-header">📱 Klasifikasi Kinerja Smartphone</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/smartphone.png", width=80)
    st.markdown("### 🎯 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini membantu mengklasifikasikan smartphone berdasarkan:
    
    **📊 Kelas Harga:**
    - Entry Level (< ₹10K)
    - Mid Range (₹10K-25K)
    - High End (₹25K-50K)
    - Flagship (> ₹50K)
    
    **🎮 Jenis Penggunaan:**
    - Gaming
    - Daily Use
    - Photography
    - Business
    - All-Rounder
    """)

# Check if models loaded
if price_model is None or usage_model is None:
    st.error("⚠️ Gagal memuat model. Pastikan notebook sudah dijalankan untuk melatih model.")
    st.info("Jalankan notebook `smartphone_performance_classification.ipynb` terlebih dahulu.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Visualisasi Data", "ℹ️ Informasi Model"])

# Tab 1: Prediction
with tab1:
    st.markdown('<p class="sub-header">Masukkan Spesifikasi Smartphone</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚡ Processor**")
        num_cores = st.slider("Jumlah Core", 4, 8, 8)
        processor_speed = st.slider("Kecepatan (GHz)", 1.0, 3.5, 2.5, 0.1)
        
        st.markdown("**💾 Memory**")
        ram_capacity = st.slider("RAM (GB)", 2, 18, 8)
        internal_memory = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024], index=2)
    
    with col2:
        st.markdown("**🔋 Battery & Display**")
        battery_capacity = st.slider("Battery (mAh)", 2000, 7000, 5000, 100)
        screen_size = st.slider("Ukuran Layar (inch)", 5.0, 8.0, 6.5, 0.1)
        refresh_rate = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144, 165], index=2)
        
        st.markdown("**📷 Camera**")
        primary_camera_rear = st.slider("Kamera Belakang (MP)", 8, 200, 50)
        primary_camera_front = st.slider("Kamera Depan (MP)", 5, 60, 16)
    
    with col3:
        st.markdown("**📐 Resolution**")
        resolution_width = st.selectbox("Width (px)", [720, 1080, 1440], index=1)
        resolution_height = st.selectbox("Height (px)", [1600, 2400, 3200], index=1)
        
        st.markdown("**✨ Features**")
        has_5g = st.checkbox("5G Support", value=True)
        has_nfc = st.checkbox("NFC", value=True)
        fast_charging = st.checkbox("Fast Charging", value=True)
        num_rear_cameras = st.slider("Jumlah Kamera Belakang", 1, 4, 3)
    
    # Predict button
    if st.button("🔍 Prediksi Klasifikasi", type="primary", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'num_cores': [num_cores],
            'processor_speed': [processor_speed],
            'ram_capacity': [ram_capacity],
            'internal_memory': [internal_memory],
            'battery_capacity': [battery_capacity],
            'screen_size': [screen_size],
            'refresh_rate': [refresh_rate],
            'primary_camera_rear': [primary_camera_rear],
            'primary_camera_front': [primary_camera_front],
            'resolution_width': [resolution_width],
            'resolution_height': [resolution_height],
            'has_5g': [int(has_5g)],
            'has_nfc': [int(has_nfc)],
            'fast_charging_available': [int(fast_charging)],
            'num_rear_cameras': [num_rear_cameras]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        price_pred = price_model.predict(input_scaled)[0]
        price_proba = price_model.predict_proba(input_scaled)[0]
        
        usage_pred = usage_model.predict(input_scaled)[0]
        usage_proba = usage_model.predict_proba(input_scaled)[0]
        
        price_class_name = price_class_names[price_pred]
        usage_type_name = usage_encoder.classes_[usage_pred]
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Hasil Prediksi</p>', unsafe_allow_html=True)
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            # Price class card
            price_class_css = price_class_name.lower().replace(" ", "-")
            st.markdown(f"""
                <div class="prediction-card {price_class_css}">
                    <h2 style="margin:0;">💰 {price_class_name}</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                        Confidence: {price_proba[price_pred]*100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Price probability chart
            price_labels = [price_class_names[i] for i in range(4)]
            fig_price = px.bar(
                x=price_labels,
                y=price_proba * 100,
                color=price_proba,
                color_continuous_scale='Blues',
                labels={'x': 'Kelas', 'y': 'Probabilitas (%)'}
            )
            fig_price.update_layout(
                title="Probabilitas Kelas Harga",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col_result2:
            # Usage type card
            usage_css = usage_type_name.lower().replace(" ", "-").replace("-use", "")
            st.markdown(f"""
                <div class="prediction-card {usage_css}">
                    <h2 style="margin:0;">🎯 {usage_type_name}</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                        Confidence: {usage_proba[usage_pred]*100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Usage probability chart
            fig_usage = px.bar(
                x=usage_encoder.classes_,
                y=usage_proba * 100,
                color=usage_proba,
                color_continuous_scale='Greens',
                labels={'x': 'Tipe', 'y': 'Probabilitas (%)'}
            )
            fig_usage.update_layout(
                title="Probabilitas Jenis Penggunaan",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_usage, use_container_width=True)
        
        # Interpretation
        st.markdown("### 💡 Interpretasi")
        
        interpretations = {
            'Entry Level': "📱 HP ini cocok untuk kebutuhan dasar seperti telepon, SMS, dan aplikasi ringan.",
            'Mid Range': "📱 HP ini memiliki keseimbangan antara harga dan performa, cocok untuk pengguna umum.",
            'High End': "📱 HP ini memiliki spesifikasi tinggi dengan harga yang masih terjangkau.",
            'Flagship': "📱 HP ini adalah kelas premium dengan spesifikasi tertinggi di kelasnya."
        }
        
        usage_interpretations = {
            'Gaming': "🎮 HP ini optimal untuk mobile gaming dengan refresh rate tinggi dan prosesor kencang.",
            'Daily Use': "📱 HP ini cocok untuk penggunaan sehari-hari dengan battery tahan lama.",
            'Photography': "📷 HP ini cocok untuk fotografi dan content creation dengan kamera berkualitas.",
            'Business': "💼 HP ini cocok untuk produktivitas dengan NFC dan storage besar.",
            'All-Rounder': "⚡ HP ini bisa digunakan untuk berbagai keperluan dengan spesifikasi seimbang."
        }
        
        st.info(f"**Kelas Harga:** {interpretations.get(price_class_name, '')}")
        st.success(f"**Jenis Penggunaan:** {usage_interpretations.get(usage_type_name, '')}")
        
        # Combined result
        st.markdown("### 🏆 Kesimpulan")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem; 
                    border-radius: 1rem; color: white; text-align: center; font-size: 1.3rem;">
            Smartphone ini termasuk kelas <b>{price_class_name}</b> dengan keunggulan untuk <b>{usage_type_name}</b>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Visualization
with tab2:
    if df is not None:
        st.markdown('<p class="sub-header">Eksplorasi Dataset Smartphone</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution by brand
            st.markdown("#### 📊 Distribusi Harga per Brand (Top 10)")
            top_brands = df.groupby('brand_name')['price'].mean().sort_values(ascending=False).head(10)
            fig_brand = px.bar(
                x=top_brands.index,
                y=top_brands.values,
                color=top_brands.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Brand', 'y': 'Rata-rata Harga (₹)'}
            )
            fig_brand.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_brand, use_container_width=True)
        
        with col2:
            # Processor brand distribution
            st.markdown("#### ⚙️ Distribusi Prosesor")
            processor_counts = df['processor_brand'].value_counts()
            fig_proc = px.pie(
                values=processor_counts.values,
                names=processor_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_proc.update_layout(height=400)
            st.plotly_chart(fig_proc, use_container_width=True)
        
        # 3D Scatter Plot
        st.markdown("#### 🔍 Visualisasi 3D: RAM vs Processor Speed vs Price")
        
        # Create price class for coloring
        df_viz = df.copy()
        df_viz['price_class'] = pd.cut(df_viz['price'], 
                                       bins=[0, 10000, 25000, 50000, float('inf')],
                                       labels=['Entry Level', 'Mid Range', 'High End', 'Flagship'])
        
        fig_3d = px.scatter_3d(
            df_viz.dropna(subset=['ram_capacity', 'processor_speed', 'price', 'price_class']),
            x='ram_capacity',
            y='processor_speed',
            z='price',
            color='price_class',
            hover_data=['brand_name', 'model'],
            labels={
                'ram_capacity': 'RAM (GB)',
                'processor_speed': 'Speed (GHz)',
                'price': 'Price (₹)',
                'price_class': 'Kelas'
            }
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Box plots
        st.markdown("#### 📈 Distribusi Fitur per Kelas Harga")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_box1 = px.box(df_viz, x='price_class', y='ram_capacity', 
                             color='price_class', title="RAM per Kelas")
            fig_box1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box1, use_container_width=True)
        
        with col2:
            fig_box2 = px.box(df_viz, x='price_class', y='processor_speed', 
                             color='price_class', title="Processor Speed per Kelas")
            fig_box2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box2, use_container_width=True)
        
        with col3:
            fig_box3 = px.box(df_viz, x='price_class', y='refresh_rate', 
                             color='price_class', title="Refresh Rate per Kelas")
            fig_box3.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box3, use_container_width=True)
        
        # Dataset statistics
        st.markdown("#### 📋 Statistik Dataset")
        st.dataframe(df.describe().round(2), use_container_width=True)

# Tab 3: Model Info
with tab3:
    st.markdown('<p class="sub-header">Informasi Model</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Tujuan Model")
        st.markdown("""
        Model ini dirancang untuk membantu pengguna memahami klasifikasi smartphone berdasarkan:
        
        1. **Kelas Harga** - Menentukan segmen pasar HP
        2. **Jenis Penggunaan** - Menentukan kecocokan penggunaan
        
        **Manfaat:**
        - Membantu konsumen memilih HP sesuai kebutuhan
        - Memberikan insight tentang spesifikasi yang mempengaruhi harga
        - Klasifikasi otomatis berdasarkan machine learning
        """)
        
        st.markdown("### 🔧 Algoritma")
        st.markdown("""
        **Random Forest Classifier**
        
        Model ensemble yang menggunakan multiple decision trees untuk:
        - Akurasi tinggi
        - Mencegah overfitting
        - Robust terhadap outliers
        
        **Anti-Overfitting Measures:**
        - Cross-validation 5-fold
        - Limited max_depth
        - Minimum samples per leaf
        - GridSearchCV optimization
        """)
    
    with col2:
        st.markdown("### 📊 Fitur Input")
        st.markdown("""
        Model menggunakan **15 fitur** utama:
        
        | Kategori | Fitur |
        |----------|-------|
        | **Processor** | num_cores, processor_speed |
        | **Memory** | ram_capacity, internal_memory |
        | **Battery** | battery_capacity |
        | **Display** | screen_size, refresh_rate, resolution |
        | **Camera** | primary_camera_rear, primary_camera_front, num_rear_cameras |
        | **Features** | has_5g, has_nfc, fast_charging |
        """)
        
        st.markdown("### 📈 Dataset")
        st.markdown("""
        - **Sumber**: smartphones.csv
        - **Jumlah Data**: ~981 smartphones
        - **Brands**: 50+ brand berbeda
        - **Price Range**: ₹3,499 - ₹650,000
        """)
    
    st.markdown("---")
    st.markdown("### 🎓 Cara Menggunakan")
    st.markdown("""
    1. Buka tab **Prediksi**
    2. Masukkan spesifikasi smartphone yang ingin diklasifikasikan
    3. Klik tombol **Prediksi Klasifikasi**
    4. Lihat hasil prediksi kelas harga dan jenis penggunaan
    5. Baca interpretasi untuk memahami hasil
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>📱 Klasifikasi Kinerja Smartphone | Powered by Random Forest</p>
        <p>Dataset: Smartphones CSV | 981 Smartphones</p>
    </div>
""", unsafe_allow_html=True)
