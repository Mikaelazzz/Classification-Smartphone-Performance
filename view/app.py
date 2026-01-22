import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Klasifikasi Smartphone",
    page_icon="📱",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model_dir = Path(__file__).resolve().parent.parent / 'model'
        
        price_model = joblib.load(model_dir / 'smartphone_price_model.pkl')
        usage_model = joblib.load(model_dir / 'smartphone_usage_model.pkl')
        scaler = joblib.load(model_dir / 'scaler.pkl')
        features = joblib.load(model_dir / 'features.pkl')
        usage_encoder = joblib.load(model_dir / 'usage_encoder.pkl')
        price_class_names = joblib.load(model_dir / 'price_class_names.pkl')
        
        return price_model, usage_model, scaler, features, usage_encoder, price_class_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

@st.cache_data
def load_data():
    try:
        data_path = Path(__file__).resolve().parent.parent / 'dataset' / 'smartphones.csv'
        return pd.read_csv(data_path)
    except:
        return None

# Load
price_model, usage_model, scaler, features, usage_encoder, price_class_names = load_models()
df = load_data()

# Header
st.markdown('<h1 class="main-header">📱 Klasifikasi Kinerja Smartphone</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check models
if price_model is None:
    st.error("⚠️ Gagal memuat model. Pastikan notebook sudah dijalankan untuk melatih model.")
    st.info("Jalankan notebook `smartphone_performance_classification.ipynb` terlebih dahulu.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Tentang Aplikasi")
    st.markdown("""
    **Klasifikasi Harga:**
    - Entry Level (< ₹10K)
    - Mid Range (₹10K-25K)
    - High End (₹25K-50K)
    - Flagship (> ₹50K)
    
    **Jenis Penggunaan:**
    - 🎮 Gaming
    - 📱 Daily Use
    - 📷 Photography
    - 💼 Business
    - ⚡ All-Rounder
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Visualisasi", "ℹ️ Info Model"])

with tab1:
    st.subheader("Masukkan Spesifikasi Smartphone")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚡ Processor & Memory**")
        num_cores = st.slider("Cores", 4, 8, 8)
        processor_speed = st.slider("Speed (GHz)", 1.0, 3.5, 2.5, 0.1)
        ram_capacity = st.slider("RAM (GB)", 2, 18, 8)
        internal_memory = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512], index=2)
    
    with col2:
        st.markdown("**🔋 Display & Battery**")
        battery_capacity = st.slider("Battery (mAh)", 2000, 7000, 5000, 100)
        screen_size = st.slider("Screen (inch)", 5.0, 8.0, 6.5, 0.1)
        refresh_rate = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144, 165], index=2)
        resolution_height = st.selectbox("Resolution Height", [1600, 2400, 3200], index=1)
        resolution_width = st.selectbox("Resolution Width", [720, 1080, 1440], index=1)
    
    with col3:
        st.markdown("**📷 Camera & Features**")
        primary_camera_rear = st.slider("Rear Camera (MP)", 8, 200, 50)
        primary_camera_front = st.slider("Front Camera (MP)", 5, 60, 16)
        num_rear_cameras = st.slider("Rear Cameras", 1, 4, 3)
        has_5g = st.checkbox("5G Support", value=True)
        has_nfc = st.checkbox("NFC", value=True)
        fast_charging = st.checkbox("Fast Charging", value=True)
    
    if st.button("🔍 Prediksi", type="primary", use_container_width=True):
        # Calculate derived features
        total_pixels = resolution_width * resolution_height
        camera_total = primary_camera_rear + primary_camera_front
        perf_score = ram_capacity * processor_speed * num_cores
        screen_ppi = total_pixels / (screen_size ** 2)
        
        # Build input based on selected features
        input_dict = {
            'num_cores': num_cores,
            'processor_speed': processor_speed,
            'ram_capacity': ram_capacity,
            'internal_memory': internal_memory,
            'battery_capacity': battery_capacity,
            'screen_size': screen_size,
            'refresh_rate': refresh_rate,
            'primary_camera_rear': primary_camera_rear,
            'primary_camera_front': primary_camera_front,
            'resolution_width': resolution_width,
            'resolution_height': resolution_height,
            'has_5g': int(has_5g),
            'has_nfc': int(has_nfc),
            'fast_charging_available': int(fast_charging),
            'num_rear_cameras': num_rear_cameras,
            'total_pixels': total_pixels,
            'camera_total': camera_total,
            'perf_score': perf_score,
            'screen_ppi': screen_ppi
        }
        
        # Select only the features used by model
        input_data = pd.DataFrame([[input_dict.get(f, 0) for f in features]], columns=features)
        input_scaled = scaler.transform(input_data)
        
        # Predict
        price_pred = price_model.predict(input_scaled)[0]
        price_proba = price_model.predict_proba(input_scaled)[0]
        
        usage_pred = usage_model.predict(input_scaled)[0]
        usage_proba = usage_model.predict_proba(input_scaled)[0]
        
        price_name = price_class_names[price_pred]
        usage_name = usage_encoder.classes_[usage_pred]
        
        # Display
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            css_class = price_name.lower().replace(" ", "-")
            st.markdown(f"""
            <div class="prediction-card {css_class}">
                <h2>💰 {price_name}</h2>
                <p>Confidence: {price_proba[price_pred]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig1 = px.bar(
                x=[price_class_names[i] for i in range(4)],
                y=price_proba * 100,
                color=price_proba,
                color_continuous_scale='Blues'
            )
            fig1.update_layout(title="Probabilitas Kelas Harga", height=300, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, #1abc9c, #16a085);">
                <h2>🎯 {usage_name}</h2>
                <p>Confidence: {usage_proba[usage_pred]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig2 = px.bar(
                x=usage_encoder.classes_,
                y=usage_proba * 100,
                color=usage_proba,
                color_continuous_scale='Greens'
            )
            fig2.update_layout(title="Probabilitas Jenis Penggunaan", height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Summary
        st.markdown("### 🏆 Kesimpulan")
        st.success(f"Smartphone ini termasuk kelas **{price_name}** dengan keunggulan untuk **{usage_name}**")
        
        # Price Range Estimation
        st.markdown("### 💰 Estimasi Rentang Harga")
        price_ranges = {
            'Entry Level': ('< ₹10,000', 'Rp 1.500.000 - Rp 3.000.000'),
            'Mid Range': ('₹10,000 - ₹25,000', 'Rp 3.000.000 - Rp 7.000.000'),
            'High End': ('₹25,000 - ₹50,000', 'Rp 7.000.000 - Rp 15.000.000'),
            'Flagship': ('> ₹50,000', 'Rp 15.000.000 - Rp 30.000.000+')
        }
        
        india_range, idr_range = price_ranges.get(price_name, ('N/A', 'N/A'))
        
        col_price1, col_price2 = st.columns(2)
        with col_price1:
            st.info(f"🇮🇳 **India:** {india_range}")
        with col_price2:
            st.info(f"🇮🇩 **Indonesia:** {idr_range}")
        
        # Similar Smartphones Recommendation
        st.markdown("### 📱 Rekomendasi Smartphone Serupa")
        
        if df is not None:
            # Filter by price class - calculate price thresholds
            price_thresholds = {
                'Entry Level': (0, 10000),
                'Mid Range': (10000, 25000),
                'High End': (25000, 50000),
                'Flagship': (50000, 999999)
            }
            
            min_price, max_price = price_thresholds.get(price_name, (0, 999999))
            
            # Filter phones in the same price class
            similar_df = df[(df['price'] >= min_price) & (df['price'] <= max_price)].copy()
            
            if not similar_df.empty:
                # Calculate similarity score based on specs
                similar_df['spec_score'] = (
                    abs(similar_df['ram_capacity'] - ram_capacity) * 2 +
                    abs(similar_df['battery_capacity'] - battery_capacity) / 500 +
                    abs(similar_df['primary_camera_rear'] - primary_camera_rear) / 10 +
                    abs(similar_df['internal_memory'] - internal_memory) / 64
                )
                
                # Get top 5 most similar phones
                top_similar = similar_df.nsmallest(5, 'spec_score')[['brand_name', 'model', 'price', 'ram_capacity', 'battery_capacity', 'primary_camera_rear']]
                
                # Format price to IDR
                def format_price(price_inr):
                    idr = price_inr * 190  # Approximate conversion rate
                    return f"Rp {idr:,.0f}".replace(',', '.')
                
                # Display as cards
                for idx, phone in top_similar.iterrows():
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 2, 1])
                        with col_a:
                            st.markdown(f"**📱 {phone['model']}**")
                        with col_b:
                            st.markdown(f"💵 ₹{phone['price']:,.0f} | {format_price(phone['price'])}")
                        with col_c:
                            st.markdown(f"RAM: {int(phone['ram_capacity'])}GB")
            else:
                st.info("Tidak ada smartphone serupa dalam dataset.")
        else:
            st.warning("Dataset tidak tersedia untuk rekomendasi.")

with tab2:
    if df is not None:
        st.subheader("Eksplorasi Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribusi Harga per Brand (Top 10)")
            top_brands = df.groupby('brand_name')['price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=top_brands.index, y=top_brands.values, color=top_brands.values)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribusi Prosesor")
            proc = df['processor_brand'].value_counts()
            fig = px.pie(values=proc.values, names=proc.index)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Dataset Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

with tab3:
    st.subheader("Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Algoritma")
        st.markdown("""
        **Random Forest Classifier**
        - n_estimators: 200 trees
        - max_depth: 8
        - class_weight: balanced
        """)
    
    with col2:
        st.markdown("### 📊 Fitur")
        st.markdown(f"**Total Fitur:** {len(features)}")
        for i, f in enumerate(features, 1):
            st.markdown(f"{i}. `{f}`")

# Footer
st.markdown("---")
st.markdown("<center>📱 Klasifikasi Kinerja Smartphone | Random Forest + Gradient Boosting</center>", unsafe_allow_html=True)
