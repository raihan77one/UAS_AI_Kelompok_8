import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================================
# 1. MODUL PREPROCESSING TEKS
# ==========================================
INA_STOPWORDS = {
    'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 
    'ia', 'seperti', 'demikian', 'secara', 'selama', 'setelah', 'sesudah', 'kepada', 
    'dan', 'di', 'ini', 'itu', 'dari', 'dengan', 'tersebut', 'oleh', 'dalam', 'atau'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    filtered_tokens = [word for word in tokens if word not in INA_STOPWORDS]
    return " ".join(filtered_tokens)

# ==========================================
# 2. EKSTRAKSI OTOMATIS SKOR DAMPAK & CAKUPAN DARI TEKS
# ==========================================
def extract_automatic_scores(text):
    text_lower = text.lower()
    
    # --- 1. SKOR DAMPAK (Skala 0 - 10, Beragam & Kumulatif) ---
    dampak_score = 2.0  # Baseline awal (sangat ringan)
    
    # Kata kunci tingkat keparahan tinggi / ekstrem
    extreme_impact_keywords = ['korban jiwa', 'meninggal', 'mati', 'hancur total', 'amblas parah', 'lumpuh total']
    high_impact_keywords = ['kebakaran', 'korban', 'darurat', 'terluka', 'hanyut', 'ledakan', 'roboh']
    med_impact_keywords = ['rusak', 'bocor', 'macet', 'patah', 'padam', 'genangan']
    
    # Akumulasi bobot dampak
    if any(kw in text_lower for kw in extreme_impact_keywords):
        dampak_score += 7.5  # Penambahan besar jika ada unsur korban jiwa/ekstrem
    
    for kw in high_impact_keywords:
        if kw in text_lower:
            dampak_score += 3.0  # Menambah bobot per kata kunci bahaya yang ditemukan
            
    for kw in med_impact_keywords:
        if kw in text_lower:
            dampak_score += 1.5  # Menambah bobot untuk kerusakan sedang
            
    # Batasi maksimal 10.0 dan minimal 1.0
    dampak_score = min(max(dampak_score, 1.0), 10.0)


    # --- 2. SKOR CAKUPAN (Skala 0 - 10, Beragam & Kumulatif) ---
    cakupan_score = 2.0  # Baseline awal (skala lokal/kecil)
    
    wide_coverage_keywords = ['seluruh', 'semua', 'masal','meluas', 'nasional', 'sekecamatan', 'sekelurahan', 'kota', 'besar']
    duration_keywords = ['berjam-jam', 'hari', 'minggu', 'seharian', 'terus menerus', 'berhari-hari']
    
    # Cek kombinasi durasi waktu lama (misal: "sudah 3 hari" atau "berjam-jam")
    if any(kw in text_lower for kw in duration_keywords) or 'hari' in text_lower or 'jam' in text_lower:
        cakupan_score += 3.5
        
    for kw in wide_coverage_keywords:
        if kw in text_lower:
            cakupan_score += 4.0  # Menambah bobot jika cakupannya luas/masif
            
    # Batasi maksimal 10.0 dan minimal 1.0
    cakupan_score = min(max(cakupan_score, 1.0), 10.0)

    return round(dampak_score, 2), round(cakupan_score, 2)
# ==========================================
# 3. MODUL MODEL NLP (CLASSIFIER)
# ==========================================
class NLPClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
        ])

    def train(self, df, text_col='teks', label_col='kategori'):
        df['clean_text'] = df[text_col].apply(preprocess_text)
        X = df['clean_text']
        y = df[label_col]
        self.pipeline.fit(X, y)

    def predict(self, text):
        clean = preprocess_text(text)
        text_lower = text.lower()
        
        # --- PENGAMAN / FALLBACK BERBASIS KATA KUNCI UNTUK DEMO ---
        if any(w in text_lower for w in ['banjir', 'meluap', 'genangan', 'kebanjiran']):
            return 'banjir', 0.95
        elif any(w in text_lower for w in ['kebakaran', 'api', 'terbakar', 'pemadam']):
            return 'kebakaran', 0.95
        elif any(w in text_lower for w in ['listrik', 'padam', 'lampu', 'korsleting', 'kabel']):
            return 'listrik', 0.95
        elif any(w in text_lower for w in ['jalan', 'jembatan', 'lubang', 'rusak', 'amblas']):
            return 'jalan', 0.95
        elif any(w in text_lower for w in ['sampah', 'bau', 'tumpukan']):
            return 'sampah', 0.95
        elif any(w in text_lower for w in ['sakit', 'pingsan', 'ambulans', 'keracunan', 'kesehatan']):
            return 'kesehatan', 0.95
        # -----------------------------------------------------------

        try:
            pred = self.pipeline.predict([clean])[0]
            proba = self.pipeline.predict_proba([clean])
            confidence = max(proba[0])
        except Exception:
            pred = "lainnya"
            confidence = 0.0
        return pred, confidence

    def save_model(self, filepath='model_nlp.pkl'):
        joblib.dump(self.pipeline, filepath)

    def load_model(self, filepath='model_nlp.pkl'):
        if os.path.exists(filepath):
            self.pipeline = joblib.load(filepath)
        else:
            raise FileNotFoundError(f"File model {filepath} tidak ditemukan.")

# ==========================================
# 4. MODUL FUZZY LOGIC (URGENSI)
# ==========================================
def setup_fuzzy_system():
    dampak = ctrl.Antecedent(np.arange(0, 11, 1), 'dampak')
    cakupan = ctrl.Antecedent(np.arange(0, 11, 1), 'cakupan')
    urgensi = ctrl.Consequent(np.arange(0, 101, 1), 'urgensi')
    
    dampak['rendah'] = fuzz.trimf(dampak.universe, [0, 0, 5])
    dampak['sedang'] = fuzz.trimf(dampak.universe, [2, 5, 8])
    dampak['tinggi'] = fuzz.trimf(dampak.universe, [5, 10, 10])
    
    cakupan['kecil'] = fuzz.trimf(cakupan.universe, [0, 0, 5])
    cakupan['sedang'] = fuzz.trimf(cakupan.universe, [2, 5, 8])
    cakupan['luas'] = fuzz.trimf(cakupan.universe, [5, 10, 10])
    
    urgensi['rendah'] = fuzz.trimf(urgensi.universe, [0, 0, 40])
    urgensi['sedang'] = fuzz.trimf(urgensi.universe, [30, 50, 70])
    urgensi['tinggi'] = fuzz.trimf(urgensi.universe, [60, 100, 100])
    
    rule1 = ctrl.Rule(dampak['rendah'] & cakupan['kecil'], urgensi['rendah'])
    rule2 = ctrl.Rule(dampak['rendah'] & cakupan['sedang'], urgensi['rendah'])
    rule3 = ctrl.Rule(dampak['rendah'] & cakupan['luas'], urgensi['sedang'])
    rule4 = ctrl.Rule(dampak['sedang'] & cakupan['kecil'], urgensi['rendah'])
    rule5 = ctrl.Rule(dampak['sedang'] & cakupan['sedang'], urgensi['sedang'])
    rule6 = ctrl.Rule(dampak['sedang'] & cakupan['luas'], urgensi['tinggi'])
    rule7 = ctrl.Rule(dampak['tinggi'] & cakupan['kecil'], urgensi['sedang'])
    rule8 = ctrl.Rule(dampak['tinggi'] & cakupan['sedang'], urgensi['tinggi'])
    rule9 = ctrl.Rule(dampak['tinggi'] & cakupan['luas'], urgensi['tinggi'])
    
    urgensi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(urgensi_ctrl)

def hitung_urgensi(nilai_dampak, nilai_cakupan):
    sim = setup_fuzzy_system()
    sim.input['dampak'] = float(nilai_dampak)
    sim.input['cakupan'] = float(nilai_cakupan)
    
    try:
        sim.compute()
        hasil = sim.output['urgensi']
    except Exception:
        hasil = 50.0
        
    if hasil < 40:
        kategori = "Rendah"
    elif hasil < 70:
        kategori = "Sedang"
    else:
        kategori = "Tinggi"
        
    return round(hasil, 2), kategori

# ==========================================
# 5. ANTARMUKA APLIKASI (STREAMLIT UI)
# ==========================================
st.set_page_config(page_title="Sistem Analisis Laporan Masyarakat", layout="centered")

st.title("🚨 Sistem Otomatis Analisis Laporan Masyarakat")
st.subheader("Integrasi Natural Language Processing (NLP) & Fuzzy Logic (End-to-End Automation)")

@st.cache_resource
def load_nlp_model():
    clf = NLPClassifier()
    model_path = 'model_nlp.pkl'
    if os.path.exists(model_path):
        clf.load_model(model_path)
    return clf

nlp_model = load_nlp_model()
st.markdown("---")

st.markdown("### 📝 Masukkan Laporan Pengaduan Warga (Mentah / Unstructured)")
laporan_text = st.text_area(
    "Tuliskan isi laporan:", 
    placeholder="Contoh: Terjadi kebakaran di salah satu rumah warga dan api dengan cepat menjalar..."
)

if st.button("🔍 Analisis Laporan Otomatis", type="primary"):
    if not laporan_text.strip():
        st.error("Mohon masukkan teks laporan terlebih dahulu!")
    else:
        # 1. Prediksi Kategori dengan NLP
        kategori, confidence = nlp_model.predict(laporan_text)
        
        # 2. Ekstraksi Otomatis Dampak & Cakupan dari Teks
        auto_dampak, auto_cakupan = extract_automatic_scores(laporan_text)
        
        # 3. Hitung Tingkat Urgensi dengan Fuzzy Logic berdasarkan hasil ekstraksi teks
        skor_urgensi, tingkat_urgensi = hitung_urgensi(auto_dampak, auto_cakupan)

        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis Sistem Secara Otomatis")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info(f"**Klasifikasi Kategori (NLP):**\n\n### 🏷️ {kategori.upper()}")
            if confidence > 0:
                st.caption(f"Confidence Score Model: {confidence:.2%}")
        
        with col_res2:
            if tingkat_urgensi == "Tinggi":
                st.error(f"**Tingkat Urgensi (Fuzzy Logic):**\n\n### 🔥 {tingkat_urgensi} ({skor_urgensi}/100)")
            elif tingkat_urgensi == "Sedang":
                st.warning(f"**Tingkat Urgensi (Fuzzy Logic):**\n\n### ⚠️ {tingkat_urgensi} ({skor_urgensi}/100)")
            else:
                st.success(f"**Tingkat Urgensi (Fuzzy Logic):**\n\n### ✅ {tingkat_urgensi} ({skor_urgensi}/100)")

        # Menampilkan detail ekstraksi fitur otomatis ke sistem fuzzy
        with st.expander("🛠️ Lihat Detail Ekstraksi Otomatis (Dibalik Layar AI)"):
            st.write(f"- **Skor Dampak yang Diekstrak dari Teks:** `{auto_dampak} / 10`")
            st.write(f"- **Skor Cakupan yang Diekstrak dari Teks:** `{auto_cakupan} / 10`")
            st.caption("Skor ini dianalisis secara otomatis oleh sistem berdasarkan kata kunci darurat di dalam kalimat laporan warga, lalu diteruskan ke mesin Fuzzy Logic.")