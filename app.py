import pandas as pd
import streamlit as st
from google_play_scraper import reviews, Sort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from googletrans import Translator
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if not word in stop_words]
    text_clean = ' '.join(tokens)
    return text_clean

# Function to detect economy-related keywords
def is_economy_related(text):
    economy_keywords = [
        "ekonomi", "bisnis", "usaha", "perdagangan", "keuangan", "investasi", "perekonomian", "produksi", "pasar",
        "pertumbuhan", "pengembangan", "lapangan kerja", "inflasi", "deflasi", "pajak", "sumber daya", "infrastruktur",
        "ekspor", "impor", "manufaktur", "perbankan", "pelaku usaha", "kewirausahaan", "kredit", "modal", "pendapatan",
        "pengeluaran", "anggaran", "moneter", "fiskal", "kebijakan ekonomi", "kesejahteraan", "kemiskinan", "ketimpangan",
        "pertanian", "industri", "teknologi", "globalisasi", "perdagangan internasional", "regulasi", "komoditas",
        "tenaga kerja", "produktivitas", "keseimbangan perdagangan", "pertumbuhan ekonomi", "kebijakan moneter",
        "kebijakan fiskal", "perekonomian digital", "ekonomi kreatif", "ekonomi syariah", "ekonomi hijau", "ekonomi sirkular",
        "ekonomi kolaboratif", "ekonomi pasar", "ekonomi mikro", "ekonomi makro", "ekonomi politik", "ekonomi pembangunan",
        "ekonomi regional", "ekonomi global", "ekonomi domestik", "ekonomi sosial", "ekonomi budaya", "ekonomi lingkungan",
        "ekonomi teknologi", "ekonomi industri", "ekonomi pertanian", "ekonomi jasa", "ekonomi transportasi", "ekonomi energi",
        "ekonomi keuangan", "ekonomi digital", "ekonomi kesehatan", "ekonomi pendidikan", "ekonomi pariwisata",
        "ekonomi hukum", "pemetaan", "ekonomi kreatif", "UKM", "sinergi", "pengembangan", "industri kreatif", "peta",
        "usaha kecil menengah", "kreativitas", "skill", "fashion", "kuliner", "potensi ekonomi", "sektor ekonomi",
        "data koordinat", "informasi pelaku ekonomi", "produk ekonomi", "monitoring", "analisis kinerja", "metode PIECES",
        "implementasi", "program aplikasi", "kinerja aplikasi", "pemetaan wilayah", "fitur aplikasi", "pelaku UKM",
        "kinerja sistem", "potensi ekonomi kreatif", "peta sektor ekonomi", "perhatian khusus", "pengambilan keputusan",
        "Dinas Perindustrian", "Propinsi Sumatera Selatan", "Badan Pusat Statistik", "izin usaha", "subsektor ekonomi",
        "kesejahteraan", "kemiskinan", "keseimbangan perdagangan", "kebijakan ekonomi", "waktu pelayanan",
        "layanan komunikasi akurat", "biaya operasional", "jumlah sumber daya", "kepuasan pengguna", "tingkat kepuasan",
        "kategori sangat puas", "efisiensi", "ekonomi", "PIECES framework", "produktivitas", "analisis kepuasan",
        "kualitas layanan", "respon pengguna", "kecepatan layanan", "efektivitas", "kualitas aplikasi", "stabilitas sistem",
        "kinerja aplikasi", "keandalan", "akurasi", "fleksibilitas", "keamanan", "pengalaman pengguna", "kemudahan penggunaan",
        "kesesuaian fitur", "pengelolaan data", "interaksi pengguna", "umpan balik pengguna", "penggunaan sumber daya",
        "pengeluaran konsumen", "produk domestik bruto", "GDP", "PDB", "inflasi", "deflasi", "nilai tukar", "perdagangan internasional",
        "investasi", "pengangguran", "tenaga kerja", "ekspor impor", "peningkatan layanan", "kepuasan total",
        "pengembangan produk", "manajemen biaya", "perbandingan biaya", "keuntungan ekonomi", "biaya vs. manfaat",
        "pembiayaan", "efektivitas biaya", "keuntungan", "laba", "pendapatan", "perusahaan teknologi", "pasar digital",
        "tren pasar", "inovasi teknologi", "adaptasi pasar", "strategi pemasaran", "bermanfaat", "nilai tambah", "aplikasi mobile",
        "pengembangan aplikasi", "penilaian kinerja", "evaluasi kualitas", "kepuasan konsumen", "tingkat penggunaan",
        "kemudahan akses", "kompatibilitas", "pembaruan aplikasi", "pemeliharaan sistem", "kapasitas layanan",
        "penyebaran aplikasi", "implementasi teknologi", "keberlanjutan", "manajemen risiko", "kinerja ekonomi",
        "penetrasi pasar", "adopsi teknologi", "perilaku konsumen", "preferensi pengguna", "analisis data", "pemrosesan data",
        "optimasi sistem", "integrasi sistem", "analisis biaya", "model bisnis", "ekonomi digital", "pertumbuhan ekonomi",
        "dampak ekonomi", "kebijakan ekonomi", "indikator ekonomi", "kebijakan publik", "distribusi pendapatan", "analisis pasar",
        "survei pengguna", "pengumpulan data", "evaluasi ekonomi", "laporan ekonomi", "tren ekonomi", "efisiensi biaya",
        "investasi asing", "manajemen keuangan", "analisis risiko", "strategi investasi", "pemberdayaan ekonomi",
        "pasar modal", "saham", "obligasi", "valuasi aset", "perencanaan keuangan", "pengelolaan aset", "kebijakan moneter",
        "kebijakan fiskal", "pertumbuhan pasar", "kredit", "pembiayaan usaha", "rasio keuangan", "analisis laporan keuangan",
        "pendapatan nasional", "belanja negara", "defisit anggaran", "surplus anggaran", "devaluasi", "revaluasi", "hutang publik",
        "pembayaran hutang", "keberlanjutan fiskal", "ekonomi mikro", "ekonomi makro", "ekonomi perilaku", "ekonomi kesehatan",
        "ekonomi pendidikan", "ekonomi lingkungan", "ekonomi regional", "ekonomi internasional", "ekonomi pembangunan",
        "ekonomi agrikultur", "ekonomi industri", "ekonomi digital", "ekonomi kreatif", "ekonomi sosial", "ekonomi politik",
        "ekonomi syariah", "ekonomi islam"
    ]
    
    for keyword in economy_keywords:
        if keyword in text.lower():
            return True
    return False

# Function to translate text
def translate_text(text):
    translator = Translator()
    try:
        translation = translator.translate(text, src='id', dest='en')
        return translation.text
    except Exception as e:
        return text

# Function to get sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity

# Function to convert sentiment polarity to Likert scale
def sentiment_to_likert(polarity):
    if polarity < -0.6:
        return 1  # Very Dissatisfied
    elif polarity < -0.2:
        return 2  # Dissatisfied
    elif polarity < 0.2:
        return 3  # Neutral
    elif polarity < 0.6:
        return 4  # Satisfied
    else:
        return 5  # Very Satisfied

# Streamlit UI
st.title('Sentiment Analysis and Domain Classification of App Reviews')

# Input for app ID
app_id = st.text_input('Enter the app ID')

if st.button('Submit'):
    # Crawling data
    ulasan_aplikasi, _ = reviews(app_id, lang='id', country='id', count=1000, filter_score_with=None, sort=Sort.MOST_RELEVANT)
    df = pd.DataFrame(ulasan_aplikasi)

    # Preprocessing data
    df['content_clean'] = df['content'].apply(preprocess_text)
    df['is_economy'] = df['content_clean'].apply(is_economy_related)

    # Calculate sentiment
    df['content_translated'] = df['content_clean'].apply(translate_text)
    df['sentiment'] = df['content_translated'].apply(get_sentiment)
    df['likert_scale'] = df['sentiment'].apply(sentiment_to_likert)

    st.subheader('App Reviews Data')
    st.markdown(f"Total reviews: {len(df)}")
    st.markdown(f"Economy-related reviews: {df['is_economy'].sum()}")

    # Display reviews data
    st.subheader('Reviews Data')
    st.write(df)

    # Display average Likert scale
    average_likert_scale = df['likert_scale'].mean()
    st.subheader('Average Likert Scale')
    st.write(f"{average_likert_scale:.2f}")

    # Visualize sentiment distribution
    st.subheader('Sentiment Distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='likert_scale', data=df, ax=ax, palette='viridis')
    ax.set_xlabel('Likert Scale')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Distribution of Sentiments Based on Likert Scale')
    st.pyplot(fig)
