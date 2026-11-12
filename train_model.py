import pandas as pd
from app_gabungan import NLPClassifier

print("Sedang memuat dataset...")
# Ganti dengan nama file dataset yang kamu miliki di folder proyek
try:
    df = pd.read_csv('dataset_gabungan.csv')
except:
    df = pd.read_excel('dataset_laporan_masyarakat_400.xlsx')

print("Sedang melatih model NLP...")
clf = NLPClassifier()
# Sesuaikan 'teks' dan 'kategori' dengan nama kolom di file Excel/CSV kamu
clf.train(df, text_col='teks', label_col='kategori')

# Simpan model yang sudah dilatih
clf.save_model('model_nlp.pkl')
print("✅ Selesai! Model berhasil dilatih dan disimpan sebagai model_nlp.pkl")