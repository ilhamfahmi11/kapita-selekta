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

# Domain keywords
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

performance_keywords = [
    "penggunaan", "ketersediaan", "menggunakan", "mengevaluasi", "memperhatikan", "mudah",
"cepat", "meningkatkan", "diakses", "baik", "sesuai", "dipahami", "digunakan", "pengguna", 
"dijalankan", "secara", "stabil", "merespon", "tampilan", "memuaskan", "tepat", "hasil", 
"kepuasan", "efisien", "mudah diakses", "mudah dipahami", "meningkatkan kepuasan", "cepat mudah", 
"tepat waktu", "waktu hasil", "dipahami mudah", "mudah digunakan", "digunakan mudah", "baik sesuai", 
"meningkatkan produktivitas", "pemanfaatan", "aksesibilitas", "evaluasi", "penilaian", "perhatian", 
"pertimbangan", "kemudahan", "keterjangkauan", "kecepatan", "kelancaran", "cepatnya", "peningkatan", 
"perbaikan", "peningkatan kualitas", "akses mudah", "kualitas baik", "kebaikan", "keunggulan", 
"keterbacaan", "kejelasan", "pemahaman yang baik", "relevan", "cocok", "relevansi", "punctuality", 
"ketepatan waktu", "waktu yang diperlukan untuk hasil", "pemahaman", "user", "pemakai", "user-friendly", 
"aplikasi yang mudah digunakan", "penggunaan yang mudah", "operasional", "dengan", "melalui", "secara efektif", 
"kestabilan", "keberlangsungan", "konsistensi", "responsif", "respons", "tanggapan", "antarmuka", 
"visualisasi", "kepuasan pengguna", "kepuasan pelanggan", "kesenjangan yang baik", "peningkatan efisiensi", 
"kinerja yang lebih baik", "akurat", "keakuratan", "ketepatan", "outcome", "output", "efisiensi", 
"penggunaan yang efisien", "penggunaan yang optimal", "penggunaan yang efektif", "pengalaman pengguna", 
"penggunaan yang lancar", "kehandalan", "peningkatan performa", "pengoptimalan", "keamanan", "pengelolaan yang baik", 
"pengelolaan yang efisien", "pengelolaan sumber daya", "integrasi yang baik", "interoperabilitas", 
"monitoring performa", "pengukuran performa", "penanganan kesalahan", "pengurangan downtime", 
"analisis performa", "optimisasi", "pemeliharaan", "manajemen risiko", "pengelolaan beban kerja", 
"pengurangan biaya", "analisis penggunaan", "pengembangan yang berkelanjutan", "penggunaan yang hemat energi", 
"analisis kepuasan pengguna", "pengelolaan kapasitas", "penyediaan layanan yang cepat", "pengembangan yang responsif", 
"penyesuaian yang mudah", "adaptabilitas", "kompatibilitas", "integrasi sistem", "fleksibilitas", 
"pengukuran keberhasilan", "peningkatan kinerja", "penggunaan sumber daya yang optimal", "pengelolaan kinerja", 
"peningkatan efektivitas", "pengoptimalan biaya", "pengurangan kesalahan", "pengelolaan performa", 
"analisis efisiensi", "analisis efektivitas", "peningkatan kehandalan", "pemantauan performa", 
"pengukuran kepuasan pengguna", "pengurangan waktu respon", "pengelolaan performa aplikasi", 
"penggunaan yang efisien dari sumber daya", "pengelolaan performa sistem", "peningkatan kualitas layanan", 
"analisis kepuasan pelanggan", "pengelolaan performa IT", "pengurangan kompleksitas", "pemantauan penggunaan", 
"pengelolaan beban kerja aplikasi", "peningkatan pengalaman pengguna", "peningkatan penggunaan sistem", 
"pengurangan konsumsi energi", "pemantauan kepuasan pengguna", "pengelolaan performa aplikasi", 
"peningkatan efisiensi penggunaan", "analisis performa sistem", "pengukuran kepuasan pelanggan", 
"pengelolaan performa jaringan", "pemantauan penggunaan aplikasi", "pengelolaan performa operasional", 
"pengukuran kepuasan pengguna akhir", "pengelolaan performa aplikasi", "pengelolaan performa operasional aplikasi", 
"analisis performa aplikasi", "pengukuran kepuasan pengguna akhir", "pengelolaan performa TI", 
"pengelolaan performa operasional sistem", "analisis performa pengguna akhir", "pengukuran efisiensi pengguna", 
"pengelolaan performa sistem TI", "analisis kepuasan pengguna akhir", "pengukuran performa aplikasi", 
"pengelolaan performa operasional aplikasi", "analisis performa sistem TI", "pengukuran kepuasan pengguna aplikasi", 
"pengelolaan performa operasional aplikasi", "analisis kepuasan pengguna aplikasi", "pengukuran performa pengguna akhir", 
"pengelolaan performa sistem informasi", "pengelolaan performa aplikasi perangkat lunak", "pengukuran performa operasional", 
"pengukuran kepuasan klien", "pengelolaan performa aplikasi perusahaan", "analisis performa aplikasi perangkat lunak", 
"pengukuran kepuasan pelanggan akhir", "pengelolaan performa sistem aplikasi", "pengelolaan performa sistem informasi perusahaan", 
"analisis kepuasan pelanggan akhir", "pengukuran efisiensi aplikasi", "pengelolaan performa aplikasi web", 
"pengukuran kepuasan pelanggan aplikasi", "pengelolaan performa aplikasi perangkat lunak", 
"analisis kepuasan pengguna perangkat lunak", "pengukuran performa operasional aplikasi perangkat lunak", 
"pengelolaan performa sistem TI", "pengelolaan performa operasional aplikasi perangkat lunak", 
"analisis performa sistem aplikasi", "pengukuran kepuasan pelanggan perangkat lunak", 
"pengelolaan performa operasional TI", "pengukuran kepuasan pengguna perangkat lunak", 
"pengelolaan performa sistem TI perusahaan", "analisis kepuasan pelanggan perangkat lunak", 
"pengukuran performa aplikasi", "pengelolaan performa operasional aplikasi", "analisis performa sistem TI", 
"pengukuran kepuasan pengguna aplikasi", "pengelolaan performa operasional aplikasi", "analisis kepuasan pengguna aplikasi", 
"pengukuran performa pengguna akhir", "pengelolaan performa sistem informasi", "pengelolaan performa aplikasi perangkat lunak", 
"pengukuran performa operasional", "pengukuran kepuasan klien", "pengelolaan performa aplikasi perusahaan", 
"analisis performa aplikasi perangkat lunak", "pengukuran kepuasan pelanggan akhir", "pengelolaan performa sistem aplikasi", 
"pengelolaan performa sistem informasi perusahaan", "analisis kepuasan pelanggan akhir", "pengukuran efisiensi aplikasi", 
"pengelolaan performa aplikasi web", "pengukuran kepuasan pelanggan aplikasi", "pengelolaan performa aplikasi perangkat lunak", 
"analisis kepuasan pengguna perangkat lunak", "pengukuran performa operasional aplikasi perangkat lunak", 
"pengelolaan performa sistem TI", "pengelolaan performa operasional aplikasi perangkat lunak", 
"analisis performa sistem aplikasi", "pengukuran kepuasan pelanggan perangkat lunak", 
"pengelolaan performa operasional TI", "pengukuran kepuasan pengguna perangkat lunak", 
"pengelolaan performa sistem TI perusahaan", "analisis kepuasan pelanggan perangkat lunak", 
"pengukuran performa aplikasi", "pengelolaan performa operasional aplikasi", "analisis performa sistem TI", 
"pengukuran kepuasan pengguna aplikasi", "pengelolaan performa operasional aplikasi", "analisis kepuasan pengguna aplikasi", 
"pengukuran performa pengguna akhir", "pengelolaan performa sistem informasi", "pengelolaan performa aplikasi perangkat lunak", 
"pengukuran performa operasional", "pengukuran kepuasan klien", "pengelolaan performa aplikasi perusahaan", 
"analisis performa aplikasi perangkat lunak", "pengukuran kepuasan pelanggan akhir", "pengelolaan performa sistem aplikasi", 
"pengelolaan performa sistem informasi perusahaan", "analisis kepuasan pelanggan akhir", "pengukuran efisiensi aplikasi", 
"pengelolaan performa aplikasi web", "pengukuran kepuasan pelanggan aplikasi", "pengelolaan performa aplikasi perangkat lunak", 
"analisis kepuasan pengguna perangkat lunak", "pengukuran performa operasional aplikasi perangkat lunak", 
"pengelolaan performa sistem TI", "pengelolaan performa operasional aplikasi perangkat lunak", 
"analisis performa sistem aplikasi", "pengukuran kepuasan pelanggan perangkat lunak", 
"pengelolaan performa operasional TI", "pengukuran kepuasan pengguna perangkat lunak", 
"pengelolaan performa sistem TI perusahaan", "analisis kepuasan pelanggan perangkat lunak", 
"pengukuran performa aplikasi", "pengelolaan performa operasional aplikasi", "analisis performa sistem TI", 
"pengukuran kepuasan pengguna aplikasi", "pengelolaan performa operasional aplikasi", "analisis kepuasan pengguna aplikasi", 
"pengukuran performa pengguna akhir", "pengelolaan performa sistem informasi", "pengelolaan performa aplikasi perangkat lunak", 
"pengukuran performa operasional", "pengukuran kepuasan klien", "pengelolaan performa aplikasi perusahaan", 
"analisis performa aplikasi perangkat lunak", "pengukuran kepuasan pelanggan akhir", "pengelolaan performa sistem aplikasi", 
"pengelolaan performa sistem informasi perusahaan", "analisis kepuasan pelanggan akhir", "pengukuran efisiensi aplikasi", 
"pengelolaan performa aplikasi web", "pengukuran kepuasan pelanggan aplikasi", "pengelolaan performa aplikasi perangkat lunak", 
"analisis kepuasan pengguna perangkat lunak", "pengukuran performa operasional aplikasi perangkat lunak", 
"pengelolaan performa sistem TI", "pengelolaan performa operasional aplikasi perangkat lunak", 
"analisis performa sistem aplikasi", "pengukuran kepuasan pelanggan perangkat lunak", 
"pengelolaan performa operasional TI", "pengukuran kepuasan pengguna perangkat lunak", 
"pengelolaan performa sistem TI perusahaan", "analisis kepuasan pelanggan perangkat lunak", 
"pengukuran performa aplikasi", "pengelolaan performa operasional aplikasi", "analisis performa sistem TI", 
"pengukuran kepuasan pengguna aplikasi", "pengelolaan performa operasional aplikasi", "analisis kepuasan pengguna aplikasi", 
"pengukuran performa pengguna akhir"
]

efficiency_keywords = [
    "efisiensi", "hemat", "penghematan", "optimal", "terbaik", "tercepat", "terefektif", "pengurangan", "pemberdayaan",
    "resource", "pemanfaatan", "reduksi", "pengurangan biaya", "manajemen waktu", "produktivitas", "kemampuan", "daya guna",
    "hasil maksimal", "kecepatan", "efektivitas", "pengembangan", "biaya minimal", "manajemen sumber daya", "peningkatan",
    "penurunan biaya", "pengeluaran minimal", "pemanfaatan maksimal", "konservasi", "sustainabilitas", "kelestarian",
    "pemberdayaan sumber daya", "hemat biaya", "cost-effective", "streamlining", "lean", "optimasi", "minimalis", "efisien",
    "kinerja tinggi", "pengurangan pemborosan", "zero waste", "penggunaan optimal", "penggunaan efisien", "resourceful",
    "economic", "frugal", "thrifty", "sparing", "saving", "budget", "manajemen biaya", "pengurangan pemborosan", "dengan memakai", "dengan metode", "dengan teknik", 
        "memaksimalkan", "menentukan", "mengenali", "memastikan", "mendeteksi", 
        "mengungkap", "memperhatikan", "melindungi", "menjaga", "mengayomi", "memelihara",
        "memasukkan", "memuat", "memperbaharui data", "menyisipkan", "menyimpan", "efisien", 
        "efektivitas", "produktif", "optimal", "cepat", "mudah", "hemat waktu", "penghematan", 
        "waktu respon", "kinerja tinggi", "tidak efisien", "kurang efisien", "membuang waktu", 
        "membuang sumber daya", "terlalu lama", "memperlambat", "mengurangi efisiensi", 
        "kinerja rendah", "tidak efektif", "terhambat", "kendala", "masalah efisiensi", 
        "keterlambatan", "tidak optimal", "tidak produktif", "mengganggu", "terlalu rumit", 
        "proses lambat", "kurang optimal", "tidak praktis", "boros", "keterbatasan", "kompleks", 
        "menghambat", "mengurangi produktivitas", "mengurangi performa", "tidak maksimal", 
        "kinerja buruk", "terganggu", "kelemahan", "menurun", "masalah kinerja", "tidak berat", "memudahkan", "mudah", "hemat", "optimal", "cepat", "ringan", "produktif", "optimal", 
"efektif", "efisiensi", "hemat", "pengurangan", "optimalisasi", "reduksi", "penyusutan", "meminimalkan", 
"mengurangi", "menghemat", "mengoptimalkan", "meningkatkan efisiensi", "mengurangi biaya", "mengurangi risiko", 
"mengoptimalkan sumber daya", "mengurangi waktu", "mengurangi konsumsi energi", "penghematan biaya", 
"pengurangan limbah", "mengurangi pengeluaran", "mengurangi penggunaan", "pengurangan beban", "mengurangi kerugian", 
"mengurangi penggunaan", "ringan dijalankan", "kemudahan penggunaan", "akses cepat", "antarmuka intuitif", 
"penghematan biaya", "pengeluaran minimal", "performa maksimal", "efisiensi puncak", "waktu respons", 
"eksekusi instan", "konsumsi rendah", "penggunaan minimal", "kinerja tinggi", "efisiensi tinggi", "hasil optimal", 
"pemakaian hemat", "minimisasi kerugian", "penyusutan biaya", "peningkatan performa", "penyempurnaan proses", 
"pengurangan beban", "penurunan konsumsi", "pengurangan ukuran", "mengurangi risiko", "penghematan energi", 
"memaksimalkan kinerja", "memperbaiki efektivitas", "peningkatan produktivitas", "pengelolaan optimal", 
"efisiensi alokasi", "pemanfaatan maksimal", "percepatan proses", "pengurangan durasi", "efisiensi daya", 
"penghematan energi", "efisiensi finansial", "meminimalisir risiko", "menghindari kerugian", "pemakaian minimal", 
"pengurangan konsumsi", "hemat energi", "pengurangan daya", "konsumsi rendah", "efisien waktu", "kerja cepat", 
"output maksimal", "penghematan biaya", "pengurangan biaya", "biaya minimal", "proses cepat", "langkah optimal", 
"prosedur efisien", "performa tinggi", "kerja ringan", "penggunaan optimal", "konsumsi minimal", "penggunaan efisien", 
"alokasi optimal", "manajemen sumber daya", "efisiensi sumber daya", "waktu cepat", "respon cepat", "akses cepat", 
"optimalisasi proses", "pemanfaatan maksimal", "optimalisasi biaya", "risiko minimal", "pengurangan risiko", 
"meminimalkan risiko", "pemborosan rendah", "beban ringan", "ekonomis", "biaya efektif", "manajemen biaya", 
"manajemen waktu", "pengurangan energi", "daya guna tinggi", "pemasukan maksimal", "sumber daya minimal", 
"proses efisien", "kerja efisien", "beban kerja rendah", "performa optimal", "biaya operasional rendah", 
"optimalisasi waktu", "efisiensi proses", "pengeluaran minimal", "pemanfaatan sumber daya", "proses lancar", 
"operasional hemat", "waktu operasional minimal", "hasil maksimal", "biaya rendah", "kinerja efisien", 
"peningkatan efisiensi", "sumber daya efektif", "daya guna maksimal", "pengurangan pemborosan", "hemat biaya", 
"efisien energi", "pengurangan waktu", "pengelolaan waktu", "pemakaian optimal", "penggunaan hemat", 
"optimalisasi energi", "waktu proses minimal", "daya guna tinggi", "penghematan maksimal", "biaya minimal", 
"efisiensi kerja", "daya guna maksimal", "pengurangan biaya operasional", "optimalisasi kinerja", "kerja optimal", 
"proses hemat", "efisiensi operasional", "pengelolaan efisien", "pengurangan pemborosan", "hasil efisien"
]

information_keywords = [
    "informasi", "sistem", "data", "database", "pengetahuan", "intelijen", "pengumpulan data", "pelaporan", "analisis data", "statistik",
    "informasi pasar", "trending", "big data", "data mining", "business intelligence", "penyimpanan data", "arsip", "metadata",
    "pengolahan data", "pengelolaan data", "kualitas data", "keamanan data", "integritas data", "informasi pengguna",
    "laporan pengguna", "feedback", "dashboard", "visualisasi data", "data analitik", "data science", "machine learning",
    "AI", "kecerdasan buatan", "algoritma", "penyajian data", "real-time data", "data sharing", "open data", "API",
    "endpoints", "data warehouse", "data lake", "katalog data", "repository", "data governance", "regulasi data",
    "kebijakan data", "hak akses data", "data compliance", "data privacy", "GDPR", "informasi teknis", "laporan teknis"
]

service_keywords = [
    "layanan", "support", "dukungan", "customer service", "pelayanan", "service level", "SLA", "waktu tanggap", "respon cepat",
    "helpdesk", "customer care", "pengalaman pelanggan", "service excellence", "kepuasan pelanggan", "klien", "pengguna",
    "service delivery", "pengelolaan layanan", "solusi", "problem solving", "perbaikan", "service desk", "call center",
    "tiket layanan", "escalation", "complaint handling", "manajemen layanan", "perawatan", "preventive maintenance",
    "troubleshooting", "support ticket", "service request", "issue resolution", "service quality", "service availability",
    "service continuity", "user support", "service cost", "service efficiency", "service reliability", "service assurance",
    "feedback loop", "customer feedback", "support team", "client support", "customer interaction", "after-sales service", "sistem", "informasi", "cepat", "waktu", "data", "perintah", "dilakukan", "merespon", "merespons", 
"sistem informasi", "menghasilkan informasi", "kinerja sistem", "perintah pembatalan", "mudah diakses", 
"sejumlah perintah", "informasi tetap", "dilakukan dengan cepat", "waktu yang dibutuhkan", 
"cepat merespon perintah", "Sistem informasi perpustakaan", "sistem informasi akuntansi", 
"kinerja sistem informasi", "diproses sistem informasi", "fungsi sistem informasi"

]

control_keywords = [
    "kontrol", "pengawasan", "monitoring", "regulasi", "standar", "kebijakan", "governance", "audit", "kepatuhan", "compliance",
    "risk management", "manajemen risiko", "quality control", "QC", "QA", "quality assurance", "internal control", "protokol",
    "procedure", "SOP", "standard operating procedure", "evaluasi", "pengukuran", "assurance", "kontrol kualitas",
    "performance control", "regulatory", "compliance check", "compliance audit", "pengawasan internal", "internal audit",
    "external audit", "risk assessment", "pengelolaan risiko", "mitigasi risiko", "governance framework", "policy enforcement",
    "kontrol akses", "access control", "security control", "pengendalian", "command", "supervision", "oversight",
    "inspection", "supervisory", "checks and balances", "reporting", "review", "assessment", "pengendalian kualitas",
    "compliance management", "monitoring system", "control system", "operational control", "financial control", "control processes",
    "policy compliance", "control framework", "regulatory compliance", "compliance monitoring", "control environment", "control activities",
    "internal control system", "management control", "audit trail", "control mechanisms", "internal governance", "control measures",
    "control evaluation", "control standards", "process control", "control guidelines", "quality management", "control procedures",
    "operational audit", "strategic control", "risk control", "control protocols", "quality oversight", "internal review",
    "proses pengawasan", "compliance reporting", "legal compliance", "audit compliance", "regulatory framework", "control implementation",
    "control policy", "control assessment", "monitoring framework", "control audit", "prosedur kontrol", "control compliance",
    "compliance strategy", "risk governance", "integrated control", "compliance framework", "data governance", "IT governance",
    "security governance", "control management"
]

# Function to detect domain-related keywords
def is_domain_related(text, domain_keywords):
    for keyword in domain_keywords:
        if keyword in text.lower():
            return True
    return False

# Function to detect multiple domains
def detect_domains(text):
    domains = []
    if is_domain_related(text, economy_keywords):
        domains.append('Economy')
    if is_domain_related(text, performance_keywords):
        domains.append('Performance')
    if is_domain_related(text, efficiency_keywords):
        domains.append('Efficiency')
    if is_domain_related(text, information_keywords):
        domains.append('Information')
    if is_domain_related(text, service_keywords):
        domains.append('Service')
    if is_domain_related(text, control_keywords):
        domains.append('Control')
    return domains

# Function to translate text
translator = Translator()
def translate_text(text):
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except:
        return text

# Function to get sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to convert sentiment to Likert scale
def sentiment_to_likert(sentiment):
    if sentiment > 0.5:
        return 5
    elif sentiment > 0.1:
        return 4
    elif sentiment >= -0.1:
        return 3
    elif sentiment >= -0.5:
        return 2
    else:
        return 1

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
    df['domains'] = df['content_clean'].apply(detect_domains)

    # Calculate sentiment
    df['content_translated'] = df['content_clean'].apply(translate_text)
    df['sentiment'] = df['content_translated'].apply(get_sentiment)
    df['likert_scale'] = df['sentiment'].apply(sentiment_to_likert)

    # Display reviews data
    st.subheader('App Reviews Data')
    st.markdown(f"Total reviews: {len(df)}")
    for domain in ['Economy', 'Performance', 'Efficiency', 'Information', 'Service', 'Control']:
        st.markdown(f"{domain}-related reviews: {df['domains'].apply(lambda x: domain in x).sum()}")

    # Display reviews data by domain
    for domain in ['Economy', 'Performance', 'Efficiency', 'Information', 'Service', 'Control']:
        domain_df = df[df['domains'].apply(lambda x: domain in x)]
        st.subheader(f'Reviews related to {domain}')
        st.write(domain_df)

        # Display average Likert scale for the domain
        average_likert_scale = domain_df['likert_scale'].mean()
        st.subheader(f'Average Likert Scale for {domain}')
        st.write(f"{average_likert_scale:.2f}")

        # Visualize sentiment distribution for the domain
        st.subheader(f'Sentiment Distribution for {domain}')
        fig, ax = plt.subplots()
        sns.countplot(x='likert_scale', data=domain_df, ax=ax, palette='viridis')
        ax.set_xlabel('Likert Scale')
        ax.set_ylabel('Number of Reviews')
        ax.set_title(f'Distribution of Sentiments Based on Likert Scale for {domain}')
        st.pyplot(fig)
