# ============================================
# STREAMLIT DASHBOARD – RUKO & RUMAH TINGGAL
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Dashboard Prediksi Harga Tanah",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_ruko_models():
    return {
        "Low": joblib.load("model_Low (2).joblib"),
        "Medium": joblib.load("model_Medium.joblib"),
        "High": joblib.load("model_High (2).joblib")
    }

@st.cache_resource
def load_rumah_model():
    model = CatBoostRegressor()
    model.load_model("catboost_final.cbm")
    return model

ruko_models = load_ruko_models()
rumah_model = load_rumah_model()

# ============================================
# SIDEBAR – PILIH JENIS PROPERTI
# ============================================

st.sidebar.title("Pengaturan Model")

jenis_properti = st.sidebar.selectbox(
    "Pilih Jenis Properti",
    ["Ruko", "Rumah Tinggal"]
)

# ============================================
# PILIH MODEL RUKO
# ============================================

if jenis_properti == "Ruko":
    model_name = st.sidebar.selectbox(
        "Pilih Segment Model Ruko",
        list(ruko_models.keys())
    )
    model = ruko_models[model_name]
    model_title = f"Gradient Boosting Regressor – {model_name}"

else:
    model = rumah_model
    model_title = "CatBoost Regressor – Rumah Tinggal"

# ============================================
# AMBIL FITUR (SUDAH OHE)
# ============================================

if jenis_properti == "Ruko":
    feature_names = list(model.feature_names_in_)
else:
    feature_names = model.feature_names_

# ============================================
# INPUT FITUR
# ============================================


st.title("Dashboard Prediksi Harga Tanah per m²")
st.subheader(model_title)
st.markdown("---")

cols = st.columns(3)
inputs = {}

for i, col in enumerate(feature_names):
    with cols[i % 3]:
        inputs[col] = st.number_input(
            col,
            value=0.0,
            step=1.0
        )

X_input = pd.DataFrame([inputs])

# ============================================
# PREDIKSI
# ============================================

if st.button("🔮 Prediksi Harga"):
    y_log = model.predict(X_input)[0]
    y_pred = np.exp(y_log)

    st.success("Prediksi berhasil!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("log(Harga)", f"{y_log:,.4f}")

    with col2:
        st.metric("Harga per m² (Rp)", f"{y_pred:,.0f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

st.markdown("---")
st.subheader("Feature Importance")

if jenis_properti == "Ruko":
    importance = model.feature_importances_
else:
    importance = model.get_feature_importance()

fi_df = pd.DataFrame({
    "Variabel": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.dataframe(fi_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(fi_df["Variabel"], fi_df["Importance"])
ax.invert_yaxis()
ax.set_title("Feature Importance")
st.pyplot(fig)

st.markdown("""
### 📝 Interpretasi Bobot Variabel
Bobot variabel menunjukkan seberapa besar kontribusi relatif setiap variabel dalam mempengaruhi prediksi harga tanah per meter persegi.
Variabel dengan bobot tertinggi merupakan faktor yang paling dominan dalam pengambilan keputusan model.
""")

feature_dictionary = {
    # =========================
    # VARIABEL NUMERIK
    # =========================
    "luas_tanah": {
        "tipe": "Numerik",
        "keterangan": "Luas tanah dalam satuan meter persegi.",
        "interpretasi": "Semakin luas tanah, semakin besar potensi nilai properti."
    },
    "luas_bangunan": {
        "tipe": "Numerik",
        "keterangan": "Luas bangunan dalam meter persegi.",
        "interpretasi": "Merepresentasikan kapasitas fisik bangunan."
    },
    "jarak_pusat_kota": {
        "tipe": "Numerik",
        "keterangan": "Jarak lokasi properti ke pusat kota (km).",
        "interpretasi": "Mewakili tingkat aksesibilitas lokasi."
    },

    # =========================
    # ONE HOT ENCODING
    # =========================
    "zona_komersial_1": {
        "tipe": "Kategorik (OHE)",
        "keterangan": "Lokasi berada di zona komersial.",
        "interpretasi": "Dibandingkan dengan zona non-komersial (baseline)."
    },
    "zona_perumahan_1": {
        "tipe": "Kategorik (OHE)",
        "keterangan": "Lokasi berada di zona perumahan.",
        "interpretasi": "Efek relatif terhadap zona referensi."
    },
    "akses_jalan_aspal": {
        "tipe": "Kategorik (OHE)",
        "keterangan": "Akses utama menggunakan jalan aspal.",
        "interpretasi": "Menunjukkan kualitas infrastruktur jalan."
    },
    "akses_jalan_beton": {
        "tipe": "Kategorik (OHE)",
        "keterangan": "Akses utama menggunakan jalan beton.",
        "interpretasi": "Dibandingkan jenis jalan lainnya."
    },
    "status_hak_SHM": {
        "tipe": "Kategorik (OHE)",
        "keterangan": "Status kepemilikan Sertifikat Hak Milik.",
        "interpretasi": "Menunjukkan kepastian hukum kepemilikan."
    }
}

fi_df["Tipe"] = fi_df["Variabel"].apply(
    lambda x: feature_dictionary.get(x, {}).get("tipe", "Lainnya")
)

fi_df["Keterangan"] = fi_df["Variabel"].apply(
    lambda x: feature_dictionary.get(x, {}).get("keterangan", "-")
)

fi_df["Interpretasi"] = fi_df["Variabel"].apply(
    lambda x: feature_dictionary.get(x, {}).get("interpretasi", "-")
)

st.subheader("📌 Bobot Variabel dan Keterangan")

st.dataframe(
    fi_df,
    use_container_width=True
)

top_vars = fi_df.head(5)

narasi = "<b>Variabel paling berpengaruh:</b><ul>"
for _, r in top_vars.iterrows():
    narasi += f"<li><b>{r['Variabel']}</b>: {r['Interpretasi']}</li>"
narasi += "</ul>"

st.markdown(narasi, unsafe_allow_html=True)


