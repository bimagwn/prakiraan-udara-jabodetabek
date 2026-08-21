"""
============================================================
Prakiraan Kualitas Udara JABODETABEK - Tampilan Publik v4
Portal informasi publik - versi sendiri.
Referensi desain: BMKG (peta + legenda + grafik peringkat),
nafas (kartu titik pantau), IQAir (panel per-polutan).

Fitur: peta interaktif 15 titik pantau, prakiraan per kota &
titik pantau, riwayat 7 hari / 30 hari / 3 bulan / 1 tahun,
panel 6 polutan (PM2.5, PM10, SO2, CO, O3, NO2),
rekomendasi aktivitas, edukasi bahasa awam.

Algoritma : Random Forest Regressor (di balik layar)
Cakupan   : 6 kota / 15 titik pantau, 2020 - Jun 2026

Penulisan Ilmiah 2026 - Bima Gunawan (50423276)
Universitas Gunadarma - Prodi Informatika
============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from pathlib import Path

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prakiraan Kualitas Udara Jabodetabek",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD MODEL & DATA (di balik layar)
# ============================================================
@st.cache_resource
def load_resources():
    model = joblib.load(BASE_DIR / 'model' / 'random_forest_air_quality.pkl')
    le_stasiun = joblib.load(BASE_DIR / 'model' / 'encoder_stasiun.pkl')
    le_kota = joblib.load(BASE_DIR / 'model' / 'encoder_kota.pkl')
    fitur = joblib.load(BASE_DIR / 'model' / 'feature_names.pkl')
    df = pd.read_parquet(BASE_DIR / 'dataset' / 'ispu_processed.parquet')
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    return model, le_stasiun, le_kota, fitur, df


model, le_st, le_kt, FITUR, df = load_resources()

TGL_DATA_AKHIR = df['tanggal'].max()          # hari pemantauan terakhir
TGL_PRED_MIN = (TGL_DATA_AKHIR + pd.Timedelta(days=1)).date()

KOTA_URUT = ['Jakarta', 'Bogor', 'Depok', 'Tangerang', 'Tangsel', 'Bekasi']
NAMA_TAMPIL = {'Tangsel': 'Tangerang Selatan'}

# Titik pantau -> WILAYAH + koordinat (lokasi publik faktual)
# Penamaan SERAGAM "Kota - Lokasi Stasiun" agar konsisten di seluruh
# tampilan dan sama dengan Tabel 3.1 naskah.
WILAYAH = {
    'DKI1': {'wilayah': 'Jakarta - Bundaran HI',    'lat': -6.1953, 'lon': 106.8231},
    'DKI2': {'wilayah': 'Jakarta - Kelapa Gading',  'lat': -6.1571, 'lon': 106.9055},
    'DKI3': {'wilayah': 'Jakarta - Jagakarsa',      'lat': -6.3369, 'lon': 106.8206},
    'DKI4': {'wilayah': 'Jakarta - Lubang Buaya',   'lat': -6.2891, 'lon': 106.9051},
    'DKI5': {'wilayah': 'Jakarta - Kebon Jeruk',    'lat': -6.1970, 'lon': 106.7743},
    'BGR1': {'wilayah': 'Bogor - Bogor Tengah',     'lat': -6.5950, 'lon': 106.7990},
    'BGR2': {'wilayah': 'Bogor - Bogor Selatan',    'lat': -6.6320, 'lon': 106.8000},
    'DPK1': {'wilayah': 'Depok - Margonda',         'lat': -6.3930, 'lon': 106.8227},
    'DPK2': {'wilayah': 'Depok - Cinere',           'lat': -6.3311, 'lon': 106.7745},
    'TGR1': {'wilayah': 'Tangerang - Cikokol',      'lat': -6.1783, 'lon': 106.6319},
    'TGR2': {'wilayah': 'Tangerang - Karawaci',     'lat': -6.2249, 'lon': 106.6060},
    'TGS1': {'wilayah': 'Tangsel - BSD City',       'lat': -6.3019, 'lon': 106.6527},
    'TGS2': {'wilayah': 'Tangsel - Pamulang',       'lat': -6.3427, 'lon': 106.7383},
    'BKS1': {'wilayah': 'Bekasi - Bekasi Kota',     'lat': -6.2383, 'lon': 106.9756},
    'BKS2': {'wilayah': 'Bekasi - Bekasi Selatan',  'lat': -6.2664, 'lon': 106.9896},
}

BULAN_ID = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
NAMA_HARI = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']


# ============================================================
# KATEGORI ISPU (warna resmi Permen LHK No. 14/2020)
# ============================================================
WARNA = {
    'BAIK':               {'bg': '#1E9E4B', 'teks': '#FFFFFF', 'muka': '😊'},
    'SEDANG':             {'bg': '#1F6FD6', 'teks': '#FFFFFF', 'muka': '🙂'},
    'TIDAK SEHAT':        {'bg': '#F2C500', 'teks': '#4A3B00', 'muka': '😷'},
    'SANGAT TIDAK SEHAT': {'bg': '#E03131', 'teks': '#FFFFFF', 'muka': '🤢'},
    'BERBAHAYA':          {'bg': '#1A1A1A', 'teks': '#FFFFFF', 'muka': '🚨'},
}
RENTANG = {'BAIK': '0 – 15,5', 'SEDANG': '15,6 – 55,4', 'TIDAK SEHAT': '55,5 – 150,4',
           'SANGAT TIDAK SEHAT': '150,5 – 250,4', 'BERBAHAYA': '> 250,4'}

DESKRIPSI = {
    'BAIK': "Udara segar, tidak berisiko bagi kesehatan.",
    'SEDANG': "Masih aman untuk umum. Kelompok sensitif sebaiknya mengurangi aktivitas luar yang lama.",
    'TIDAK SEHAT': "Mulai berisiko. Kurangi aktivitas di luar, gunakan masker bila keluar.",
    'SANGAT TIDAK SEHAT': "Berisiko tinggi bagi semua orang. Hindari aktivitas di luar ruang.",
    'BERBAHAYA': "Kondisi darurat. Tetap di dalam ruangan.",
}


# Konfigurasi grafik: tanpa bilah alat, tanpa perbesar-gulir, dan
# klik-ganda mengembalikan tampilan semula (jaring pengaman di layar sentuh).
KONFIG_GRAFIK = {'displayModeBar': False, 'scrollZoom': False,
                 'doubleClick': 'reset'}


def warna_teks(kat):
    """Warna teks angka di atas kartu putih (kuning perlu digelapkan)."""
    return WARNA[kat]['bg'] if kat != 'TIDAK SEHAT' else '#B58F00'


def kategori_pm25(pm25):
    """Kategori berdasarkan breakpoint PM2.5 Permen LHK 14/2020."""
    if pm25 <= 15.5:  return 'BAIK'
    if pm25 <= 55.4:  return 'SEDANG'
    if pm25 <= 150.4: return 'TIDAK SEHAT'
    if pm25 <= 250.4: return 'SANGAT TIDAK SEHAT'
    return 'BERBAHAYA'


# --- Indeks ISPU per polutan (breakpoint Permen LHK 14/2020) ---
def pm25_to_ispu(c):
    if c <= 15.5:  return np.interp(c, [0, 15.5], [0, 50])
    if c <= 55.4:  return np.interp(c, [15.6, 55.4], [51, 100])
    if c <= 150.4: return np.interp(c, [55.5, 150.4], [101, 200])
    if c <= 250.4: return np.interp(c, [150.5, 250.4], [201, 300])
    return min(500, np.interp(c, [250.5, 500], [301, 500]))

def pm10_to_ispu(c):
    if c <= 50:  return np.interp(c, [0, 50], [0, 50])
    if c <= 150: return np.interp(c, [51, 150], [51, 100])
    if c <= 350: return np.interp(c, [151, 350], [101, 200])
    if c <= 420: return np.interp(c, [351, 420], [201, 300])
    return min(500, np.interp(c, [421, 600], [301, 500]))

def so2_to_ispu(c):
    if c <= 52:  return np.interp(c, [0, 52], [0, 50])
    if c <= 180: return np.interp(c, [53, 180], [51, 100])
    if c <= 400: return np.interp(c, [181, 400], [101, 200])
    if c <= 800: return np.interp(c, [401, 800], [201, 300])
    return min(500, np.interp(c, [801, 1200], [301, 500]))

def co_to_ispu(c):
    if c <= 4:  return np.interp(c, [0, 4], [0, 50])
    if c <= 8:  return np.interp(c, [4.1, 8], [51, 100])
    if c <= 15: return np.interp(c, [8.1, 15], [101, 200])
    if c <= 30: return np.interp(c, [15.1, 30], [201, 300])
    return min(500, np.interp(c, [30.1, 60], [301, 500]))

def o3_to_ispu(c):
    if c <= 120: return np.interp(c, [0, 120], [0, 50])
    if c <= 235: return np.interp(c, [121, 235], [51, 100])
    if c <= 400: return np.interp(c, [236, 400], [101, 200])
    if c <= 800: return np.interp(c, [401, 800], [201, 300])
    return min(500, np.interp(c, [801, 1200], [301, 500]))

def no2_to_ispu(c):
    if c <= 80:   return np.interp(c, [0, 80], [0, 50])
    if c <= 200:  return np.interp(c, [81, 200], [51, 100])
    if c <= 1130: return np.interp(c, [201, 1130], [101, 200])
    if c <= 2260: return np.interp(c, [1131, 2260], [201, 300])
    return min(500, np.interp(c, [2261, 3000], [301, 500]))


def kategori_ispu(nilai):
    if nilai <= 50:  return 'BAIK'
    if nilai <= 100: return 'SEDANG'
    if nilai <= 200: return 'TIDAK SEHAT'
    if nilai <= 300: return 'SANGAT TIDAK SEHAT'
    return 'BERBAHAYA'


POLUTAN_INFO = {
    'pm25': {'nama': 'PM2.5', 'satuan': 'µg/m³', 'ispu': pm25_to_ispu},
    'pm10': {'nama': 'PM10',  'satuan': 'µg/m³', 'ispu': pm10_to_ispu},
    'so2':  {'nama': 'SO₂',   'satuan': 'µg/m³', 'ispu': so2_to_ispu},
    'co':   {'nama': 'CO',    'satuan': 'mg/m³', 'ispu': co_to_ispu},
    'o3':   {'nama': 'O₃',    'satuan': 'µg/m³', 'ispu': o3_to_ispu},
    'no2':  {'nama': 'NO₂',   'satuan': 'µg/m³', 'ispu': no2_to_ispu},
}


def rekomendasi(kategori):
    """Rekomendasi aktivitas untuk masyarakat awam."""
    r = {
        'BAIK': [
            ("🏃", "Olahraga & aktivitas luar ruang aman untuk semua"),
            ("🧒", "Anak-anak boleh bermain di luar"),
            ("🪟", "Silakan buka jendela untuk sirkulasi udara"),
            ("👕", "Menjemur pakaian di luar aman"),
        ],
        'SEDANG': [
            ("🚶", "Aktivitas luar ruang masih aman untuk umum"),
            ("⚠️", "Penderita asma, ibu hamil & lansia: batasi aktivitas luar yang lama"),
            ("🏃", "Olahraga ringan di luar masih boleh"),
            ("🪟", "Jendela boleh dibuka, pantau kondisi"),
        ],
        'TIDAK SEHAT': [
            ("😷", "Gunakan masker (minimal KN95) saat keluar rumah"),
            ("🏃", "Hindari olahraga berat di luar ruang"),
            ("🧒", "Anak-anak, lansia & penderita asma: tetap di dalam"),
            ("🪟", "Tutup jendela; gunakan penjernih udara bila ada"),
        ],
        'SANGAT TIDAK SEHAT': [
            ("🏠", "Hindari semua aktivitas luar ruang"),
            ("😷", "Wajib masker N95/KN95 bila terpaksa keluar"),
            ("🪟", "Tutup rapat jendela & pintu"),
            ("🏥", "Segera periksa bila muncul sesak napas / batuk"),
        ],
        'BERBAHAYA': [
            ("🚨", "Kondisi darurat - tetap di dalam ruangan"),
            ("😷", "Masker N95 bahkan untuk keperluan singkat di luar"),
            ("🏥", "Hubungi fasilitas kesehatan bila ada gejala"),
            ("📻", "Ikuti arahan resmi pemerintah setempat"),
        ],
    }
    return r.get(kategori, [])


def fmt_tanggal(d):
    return f"{NAMA_HARI[d.weekday()]}, {d.day} {BULAN_ID[d.month]} {d.year}"


# ============================================================
# MESIN PRAKIRAAN (model memprediksi tiap titik pantau)
# ============================================================
@st.cache_data(show_spinner=False)
def prakiraan_semua(tanggal_iso: str) -> pd.DataFrame:
    """Prediksi PM2.5 semua titik pantau pada tanggal tertentu."""
    t = pd.Timestamp(tanggal_iso)
    rows, meta = [], []
    for stasiun in sorted(df['stasiun'].unique()):
        sub = df[df['stasiun'] == stasiun].sort_values('tanggal').tail(400)
        if len(sub) < 365 or stasiun not in WILAYAH:
            continue
        kota = sub['kota'].iloc[0]
        rows.append({
            'stasiun_id':    le_st.transform([stasiun])[0],
            'kota_id':       le_kt.transform([kota])[0],
            'tahun':         t.year,
            'bulan':         t.month,
            'hari':          t.day,
            'hari_minggu':   t.dayofweek,
            'kuartal':       t.quarter,
            'minggu_tahun':  t.isocalendar()[1],
            'hari_tahun':    t.dayofyear,
            'akhir_pekan':   1 if t.dayofweek in [5, 6] else 0,
            'musim_hujan':   1 if t.month in [11, 12, 1, 2, 3] else 0,
            'pm25_lag_1':    sub['pm25'].iloc[-1],
            'pm25_lag_3':    sub['pm25'].iloc[-3],
            'pm25_lag_7':    sub['pm25'].iloc[-7],
            'pm25_lag_14':   sub['pm25'].iloc[-14],
            'pm25_lag_30':   sub['pm25'].iloc[-30],
            'pm25_lag_60':   sub['pm25'].iloc[-60],
            'pm25_lag_90':   sub['pm25'].iloc[-90],
            'pm25_lag_180':  sub['pm25'].iloc[-180],
            'pm25_lag_365':  sub['pm25'].iloc[-365],
            'pm25_rata_7':   sub['pm25'].tail(7).mean(),
            'pm25_rata_14':  sub['pm25'].tail(14).mean(),
            'pm25_rata_30':  sub['pm25'].tail(30).mean(),
            'pm25_rata_60':  sub['pm25'].tail(60).mean(),
            'pm25_rata_90':  sub['pm25'].tail(90).mean(),
            'pm25_rata_365': sub['pm25'].tail(365).mean(),
            'pm25_std_30':   sub['pm25'].tail(30).std(),
            'pm10_lag_1':    sub['pm10'].iloc[-1],
            'so2_lag_1':     sub['so2'].iloc[-1],
            'co_lag_1':      sub['co'].iloc[-1],
            'o3_lag_1':      sub['o3'].iloc[-1],
            'no2_lag_1':     sub['no2'].iloc[-1],
        })
        meta.append({'stasiun': stasiun, 'kota': kota,
                     'wilayah': WILAYAH[stasiun]['wilayah'],
                     'lat': WILAYAH[stasiun]['lat'],
                     'lon': WILAYAH[stasiun]['lon']})
    inp = pd.DataFrame(rows)[FITUR]
    pred = model.predict(inp)
    hasil = pd.DataFrame(meta)
    hasil['pm25'] = pred
    hasil['kategori'] = hasil['pm25'].apply(kategori_pm25)
    return hasil


@st.cache_data(show_spinner=False)
def riwayat_kota(kota: str, n_hari: int):
    """Rata-rata harian PM2.5 sebuah kota, n hari terakhir."""
    sub = df[df['kota'] == kota].groupby('tanggal')['pm25'].mean().reset_index()
    return sub.tail(n_hari)


@st.cache_data(show_spinner=False)
def polutan_terakhir(kota: str):
    """Rata-rata 6 polutan sebuah kota pada hari pemantauan terakhir."""
    sub = df[(df['kota'] == kota) & (df['tanggal'] == TGL_DATA_AKHIR)]
    if sub.empty:
        sub = df[df['kota'] == kota].sort_values('tanggal').tail(4)
    return {p: sub[p].mean() for p in POLUTAN_INFO}


# ============================================================
# GAYA TAMPILAN v4 (premium + animasi halus)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    .stApp { background: #EEF3F9 !important; font-family: 'Plus Jakarta Sans', sans-serif; color: #1C2B3A; }
    h1, h2, h3, h4 { color: #12395C !important; font-family: 'Plus Jakarta Sans', sans-serif; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

    @keyframes fadeUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;} }
    @keyframes geser  { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }
    @keyframes denyut { 0%,100% {opacity:1;} 50% {opacity:0.35;} }

    .blok-judul {
        background: linear-gradient(120deg, #08365F, #0A4D8C, #1272C4, #2E9BE6, #0A4D8C);
        background-size: 300% 300%;
        animation: geser 14s ease infinite;
        border-radius: 18px; padding: 1.8rem 2.2rem; color: #FFFFFF; margin-bottom: 1.2rem;
        box-shadow: 0 8px 28px rgba(10, 77, 140, 0.25);
    }
    .blok-judul h1 { color: #FFFFFF !important; margin: 0; font-size: 1.75rem; font-weight: 800; }
    .blok-judul p  { color: #D6E9FA; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
    .chip-legenda {
        display:inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
        font-size: 0.74rem; font-weight: 700; margin: 0.5rem 0.3rem 0 0;
        background: rgba(255,255,255,0.14); color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.25); backdrop-filter: blur(4px);
    }
    .kartu {
        background: #FFFFFF; border-radius: 16px; padding: 1rem 1.1rem;
        box-shadow: 0 2px 12px rgba(18, 57, 92, 0.08); border: 1px solid #E3ECF5;
        animation: fadeUp .5s ease both;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .kartu:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(18, 57, 92, 0.14); }
    .kartu-kota { text-align: center; padding: 1.1rem 0.6rem; }
    .badge {
        display: inline-block; padding: 0.28rem 0.85rem; border-radius: 999px;
        font-weight: 700; font-size: 0.78rem; letter-spacing: 0.02em;
    }
    .angka-besar { font-size: 2.1rem; font-weight: 800; margin: 0.2rem 0 0.1rem 0; }
    .satuan { font-size: 0.8rem; color: #64798D; }
    .item-saran {
        background: #FFFFFF; border: 1px solid #E3ECF5; border-radius: 12px;
        padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.95rem;
        box-shadow: 0 1px 5px rgba(18, 57, 92, 0.05);
        animation: fadeUp .5s ease both;
        transition: transform .2s ease;
    }
    .item-saran:hover { transform: translateX(4px); }
    .kartu-polutan {
        background:#FFFFFF; border:1px solid #E3ECF5; border-radius:14px;
        padding:0.8rem 0.9rem; text-align:center;
        box-shadow: 0 1px 6px rgba(18, 57, 92, 0.06);
        animation: fadeUp .55s ease both;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .kartu-polutan:hover { transform: translateY(-3px); box-shadow: 0 8px 18px rgba(18, 57, 92, 0.12); }
    .dot-kritis { animation: denyut 1.5s ease-in-out infinite; }
    .catatan {
        background: #FFF8E6; border: 1px solid #F2E3B3; border-radius: 12px;
        padding: 0.9rem 1.2rem; font-size: 0.85rem; color: #6B5A1E;
    }
    div[data-baseweb="select"] > div { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO (dengan legenda kategori sebagai chip)
# ============================================================
chip = "".join(
    f"<span class='chip-legenda'>{w['muka']} {kat} · {RENTANG[kat]}</span>"
    for kat, w in WARNA.items())
st.markdown(f"""
<div class='blok-judul'>
    <h1>🌤️ Prakiraan Kualitas Udara Jabodetabek</h1>
    <p>Prakiraan PM2.5 harian, peta titik pantau, dan rekomendasi aktivitas luar ruang —
    Jakarta, Bogor, Depok, Tangerang, Tangerang Selatan, dan Bekasi.</p>
    <div>{chip}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PILIH TANGGAL PRAKIRAAN
# ============================================================
kol_tgl, kol_ket = st.columns([1, 2.4], gap="large")
with kol_tgl:
    # horizon dipatok 14 hari SETELAH data terakhir (bukan
    # setelah tanggal hari ini): model one-step-ahead bersandar
    # pada riwayat terakhir, sehingga makin jauh dari data
    # terakhir prakiraan makin tidak dapat dipertanggungjawabkan
    default_tgl = TGL_PRED_MIN
    tgl_maks = TGL_PRED_MIN + timedelta(days=13)
    tanggal = st.date_input(
        "📅 Tanggal prakiraan",
        value=default_tgl,
        min_value=TGL_PRED_MIN,
        max_value=tgl_maks)
with kol_ket:
    st.markdown(f"""
    <div class='kartu' style='margin-top:1.7rem; padding:0.7rem 1.1rem;'>
        Menampilkan prakiraan untuk <strong>{fmt_tanggal(tanggal)}</strong>
        &nbsp;·&nbsp; disusun dari riwayat pemantauan hingga
        <strong>{fmt_tanggal(TGL_DATA_AKHIR)}</strong>. Prakiraan paling
        andal untuk hari pertama setelah pemantauan terakhir; makin jauh
        tanggalnya, makin besar ketidakpastiannya.
    </div>
    """, unsafe_allow_html=True)

hasil = prakiraan_semua(tanggal.isoformat())
stasiun_lewat = sorted(set(df['stasiun'].unique()) - set(hasil['stasiun']))
if stasiun_lewat:
    st.warning("Titik pantau berikut tidak ditampilkan (riwayat belum "
               f"lengkap / titik pantau belum terdaftar): {', '.join(stasiun_lewat)}")
ringkas_kota = hasil.groupby('kota')['pm25'].mean().to_dict()
aktual_kemarin = (df[df['tanggal'] == TGL_DATA_AKHIR]
                  .groupby('kota')['pm25'].mean().to_dict())


# ============================================================
# RINGKASAN 6 KOTA (jawaban utama - paling atas)
# ============================================================
st.markdown("### Prakiraan PM2.5 per Kota")
st.caption("Enam kota administratif Jabodetabek. Kota Tangerang dan Kota "
           "Tangerang Selatan merupakan dua daerah otonom yang terpisah, "
           "sedangkan DKI Jakarta ditampilkan sebagai satu kesatuan dengan "
           "rincian per titik pantaunya pada bagian Detail Kota di bawah.")

k_bersih = min(ringkas_kota, key=ringkas_kota.get)
k_tinggi = max(ringkas_kota, key=ringkas_kota.get)
st.markdown(f"""
<div class='kartu' style='padding:0.65rem 1.1rem; margin-bottom:0.8rem; font-size:0.92rem;'>
    💨 Udara paling bersih diprakirakan di
    <strong>{NAMA_TAMPIL.get(k_bersih, k_bersih)}</strong> ({ringkas_kota[k_bersih]:.0f} µg/m³)
    &nbsp;·&nbsp; paling perlu diwaspadai di
    <strong>{NAMA_TAMPIL.get(k_tinggi, k_tinggi)}</strong> ({ringkas_kota[k_tinggi]:.0f} µg/m³).
</div>
""", unsafe_allow_html=True)
kolom = st.columns(6, gap="small")
for i, kota in enumerate(KOTA_URUT):
    nilai = ringkas_kota.get(kota)
    if nilai is None:
        continue
    kat = kategori_pm25(nilai)
    w = WARNA[kat]
    akt = aktual_kemarin.get(kota)
    # <= 0,5 agar selaras dengan pembulatan label :.0f (0,5 tampil "0")
    if akt is None or abs(nilai - akt) <= 0.5:
        tren_txt, tren_wrn = "≈ stabil", "#64798D"
    elif nilai > akt:
        tren_txt, tren_wrn = f"▲ {nilai - akt:.0f} dari kemarin", "#C0392B"
    else:
        tren_txt, tren_wrn = f"▼ {akt - nilai:.0f} dari kemarin", "#1E9E4B"
    with kolom[i]:
        st.markdown(f"""
        <div class='kartu kartu-kota' style='border-top: 5px solid {w["bg"]}; animation-delay: {i*0.06:.2f}s;'>
            <div style='font-weight:700; font-size:0.95rem;'>{NAMA_TAMPIL.get(kota, kota)}</div>
            <div class='angka-besar' style='color:{warna_teks(kat)};'>{nilai:.0f}</div>
            <div class='satuan'>PM2.5 (µg/m³)</div>
            <div style='font-size:0.72rem; margin-top:0.25rem; font-weight:600; color:{tren_wrn};'>{tren_txt}</div>
            <div style='margin-top:0.45rem;'>
                <span class='badge' style='background:{w["bg"]}; color:{w["teks"]};'>{w["muka"]} {kat}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# ============================================================
# PETA WILAYAH (kamera DIKUNCI ke Jabodetabek)
# ============================================================
st.markdown("### 🗺️ Peta Prakiraan PM2.5 — 15 Titik Pantau Jabodetabek")

fig_map = px.scatter_map(
    hasil, lat='lat', lon='lon',
    color='kategori',
    color_discrete_map={k: w['bg'] for k, w in WARNA.items()},
    size=np.clip(hasil['pm25'], 12, None), size_max=26,
    hover_name='wilayah',
    custom_data=['wilayah', 'pm25', 'kategori'],
    height=440)
fig_map.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>PM2.5: %{customdata[1]:.0f} µg/m³<br>Kategori: %{customdata[2]}<extra></extra>')
# kunci kamera di layout langsung (jangan andalkan argumen px,
# karena tema Streamlit bisa menimpanya saat render)
fig_map.update_layout(
    map=dict(style='carto-positron',
             center=dict(lat=-6.35, lon=106.82), zoom=8.8),
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='#FFFFFF',
    legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0,
                bgcolor='rgba(0,0,0,0)', title=None,
                font=dict(size=12, color='#33475C')))
st.plotly_chart(fig_map, width='stretch', theme=None,
                config={'displayModeBar': False})
st.caption("Ukuran & warna lingkaran mengikuti prakiraan PM2.5 tiap titik pantau. "
           "Arahkan kursor untuk detail. (Latar peta membutuhkan koneksi internet; "
           "tanpa internet, titik pantau tetap tampil.)")

st.markdown("")


# ============================================================
# GRAFIK PERINGKAT 15 WILAYAH
# ============================================================
st.markdown(f"### Prakiraan PM2.5 per Titik Pantau — {fmt_tanggal(tanggal)}")
st.caption("Seluruh 15 titik pantau di enam kota, diurutkan dari prakiraan "
           "tertinggi ke terendah.")

urut = hasil.sort_values('pm25', ascending=False)
warna_bar = [WARNA[k]['bg'] for k in urut['kategori']]
fig_rank = go.Figure(go.Bar(
    x=urut['wilayah'], y=urut['pm25'],
    marker_color=warna_bar, marker_line_width=0,
    hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.0f} µg/m³<extra></extra>'))
fig_rank.add_hline(y=55.4, line_dash='dot', line_color='#E03131',
                   annotation_text='Batas SEDANG → TIDAK SEHAT',
                   annotation_font_color='#E03131')
fig_rank.update_layout(
    plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
    font_color='#33475C', height=380,
    margin=dict(l=10, r=10, t=20, b=10),
    # automargin: sisa ruang bawah menyesuaikan panjang label miring,
    # supaya nama titik pantau tidak terpotong di layar sempit
    xaxis=dict(gridcolor='#EDF2F8', tickangle=-40,
               automargin=True, tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#EDF2F8', title='PM2.5 (µg/m³)'),
    # dragmode=False: cegah kotak-perbesar tak sengaja saat layar disentuh
    dragmode=False,
    bargap=0.35)
st.plotly_chart(fig_rank, width='stretch',
                config=KONFIG_GRAFIK)
st.caption("Warna batang mengikuti kategori ISPU. Titik pantau paling kiri = prakiraan PM2.5 tertinggi.")

st.markdown("")


# ============================================================
# DETAIL KOTA + REKOMENDASI + RIWAYAT (7H / 30H / 3B / 1T)
# ============================================================
st.markdown("### Detail Kota & Rekomendasi Aktivitas")

kota_pilih = st.selectbox(
    "Pilih kota",
    options=KOTA_URUT,
    format_func=lambda k: NAMA_TAMPIL.get(k, k))

sub_hasil = hasil[hasil['kota'] == kota_pilih].sort_values('pm25', ascending=False)
nilai_kota = sub_hasil['pm25'].mean()
kat_kota = kategori_pm25(nilai_kota)
wk = WARNA[kat_kota]

kol_kiri, kol_kanan = st.columns([1, 1.25], gap="large")

with kol_kiri:
    st.markdown(f"""
    <div class='kartu' style='border-left: 6px solid {wk["bg"]}; padding:1.4rem 1.5rem;'>
        <div style='color:#64798D; font-size:0.88rem;'>
            {NAMA_TAMPIL.get(kota_pilih, kota_pilih)} · {fmt_tanggal(tanggal)}
        </div>
        <div style='font-size:3rem; font-weight:800; color:#12395C; margin:0.2rem 0;'>
            {nilai_kota:.0f} <span style='font-size:1.05rem; color:#64798D; font-weight:600;'>µg/m³ (PM2.5)</span>
        </div>
        <span class='badge' style='background:{wk["bg"]}; color:{wk["teks"]}; font-size:0.9rem;'>{wk["muka"]} {kat_kota}</span>
        <p style='margin:0.8rem 0 0 0; color:#33475C; font-size:0.95rem;'>{DESKRIPSI[kat_kota]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown(f"**Rincian per titik pantau "
                f"{NAMA_TAMPIL.get(kota_pilih, kota_pilih)} — prakiraan PM2.5:**")
    for _, r in sub_hasil.iterrows():
        ww = WARNA[r['kategori']]
        st.markdown(f"""
        <div class='item-saran' style='display:flex; justify-content:space-between;
                    align-items:center; gap:0.75rem;'>
            <span style='font-weight:600; min-width:0;'>{r['wilayah']}</span>
            <span style='white-space:nowrap; flex-shrink:0;'>
                <strong style='margin-right:0.6rem;'>{r['pm25']:.0f} µg/m³</strong>
                <span class='badge' style='background:{ww["bg"]}; color:{ww["teks"]};'>{r['kategori']}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Apa yang sebaiknya dilakukan:**")
    for emo, teks in rekomendasi(kat_kota):
        st.markdown(f"<div class='item-saran'>{emo} &nbsp;{teks}</div>",
                    unsafe_allow_html=True)

with kol_kanan:
    st.markdown(f"**Riwayat PM2.5 {NAMA_TAMPIL.get(kota_pilih, kota_pilih)} "
                f"— data pemantauan yang sudah berlalu**")
    rentang_pilih = st.radio(
        "Rentang waktu",
        options=['7 Hari', '30 Hari', '3 Bulan', '1 Tahun'],
        index=1, horizontal=True, label_visibility='collapsed')
    N_HARI = {'7 Hari': 7, '30 Hari': 30, '3 Bulan': 90, '1 Tahun': 365}
    riw = riwayat_kota(kota_pilih, N_HARI[rentang_pilih])

    mode_garis = 'lines+markers' if len(riw) <= 45 else 'lines'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=riw['tanggal'], y=riw['pm25'],
        mode=mode_garis, name='PM2.5',
        line=dict(color='#1272C4', width=2.5),
        marker=dict(size=5),
        fill='tozeroy', fillcolor='rgba(18, 114, 196, 0.07)',
        hovertemplate='%{x|%d %b %Y}<br>PM2.5: %{y:.0f} µg/m³<extra></extra>'))
    fig.add_hline(y=15.5, line_dash='dot', line_color='#1E9E4B',
                  annotation_text='Batas BAIK', annotation_font_color='#1E9E4B')
    fig.add_hline(y=55.4, line_dash='dot', line_color='#E03131',
                  annotation_text='Batas SEDANG', annotation_font_color='#E03131')
    fig.update_layout(
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        font_color='#33475C', height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor='#EDF2F8', title=None),
        yaxis=dict(gridcolor='#EDF2F8', title='PM2.5 (µg/m³)'),
        dragmode=False,
        showlegend=False)
    st.plotly_chart(fig, width='stretch', config=KONFIG_GRAFIK)
    st.caption("Garis = rata-rata harian kota. Di bawah garis hijau = BAIK; "
               "hijau–merah = SEDANG; di atas merah = TIDAK SEHAT.")

    # ringkasan rentang terpilih (bahasa awam)
    kat_riw = riw['pm25'].apply(kategori_pm25)
    dominan = kat_riw.value_counts().idxmax()
    wd = WARNA[dominan]
    st.markdown(f"""
    <div class='kartu' style='padding:0.8rem 1.1rem;'>
        Dalam <strong>{rentang_pilih.lower()}</strong> terakhir, udara
        {NAMA_TAMPIL.get(kota_pilih, kota_pilih)} paling sering berada di kategori
        <span class='badge' style='background:{wd["bg"]}; color:{wd["teks"]};'>{dominan}</span>
        &nbsp;({(kat_riw == dominan).mean()*100:.0f}% hari) ·
        terendah {riw['pm25'].min():.0f} · tertinggi {riw['pm25'].max():.0f} µg/m³.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PANEL 6 POLUTAN
# ============================================================
nilai_pol = polutan_terakhir(kota_pilih)
st.markdown("")
st.markdown(f"### Kondisi 6 Polutan — {NAMA_TAMPIL.get(kota_pilih, kota_pilih)}")
st.caption(f"Pemantauan terakhir: {fmt_tanggal(TGL_DATA_AKHIR)}. "
           "Indeks dihitung per polutan sesuai standar ISPU; "
           "polutan dengan indeks tertinggi disebut polutan kritis.")

indeks = {p: POLUTAN_INFO[p]['ispu'](v) for p, v in nilai_pol.items()}
kritis = max(indeks, key=indeks.get)

kolom_pol = st.columns(6, gap="small")
for i, (p, info) in enumerate(POLUTAN_INFO.items()):
    v = nilai_pol[p]
    idx = indeks[p]
    kat_p = kategori_ispu(idx)
    wp = WARNA[kat_p]
    tanda_kritis = ("<div class='dot-kritis' style='font-size:0.68rem; font-weight:800; "
                    "color:#E03131; margin-top:0.3rem;'>● POLUTAN KRITIS</div>") if p == kritis else ""
    with kolom_pol[i]:
        st.markdown(f"""
        <div class='kartu-polutan' style='border-top: 4px solid {wp["bg"]};
             animation-delay: {i*0.06:.2f}s;
             {"box-shadow: 0 0 0 2px #E0313155;" if p == kritis else ""}'>
            <div style='font-weight:800; font-size:1rem;'>{info['nama']}</div>
            <div style='font-size:1.35rem; font-weight:800; color:#12395C; margin:0.15rem 0;'>
                {v:.1f}<span style='font-size:0.68rem; color:#64798D;'> {info['satuan']}</span>
            </div>
            <div style='font-size:0.7rem; color:#64798D;'>Indeks: {idx:.0f}</div>
            <span class='badge' style='background:{wp["bg"]}; color:{wp["teks"]};
                  font-size:0.62rem; margin-top:0.3rem;'>{kat_p}</span>
            {tanda_kritis}
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# ============================================================
# EDUKASI SINGKAT (bahasa awam)
# ============================================================
st.markdown("### Kenali Kualitas Udara")

kol_e1, kol_e2 = st.columns([1, 1.35], gap="large")

with kol_e1:
    with st.expander("💨 Apa itu PM2.5?", expanded=False):
        st.markdown(
            "PM2.5 adalah **partikel debu sangat halus** di udara — ukurannya "
            "sekitar **1/30 tebal sehelai rambut**. Karena sangat kecil, partikel "
            "ini bisa terhirup sampai ke bagian terdalam paru-paru, sehingga "
            "menjadi polutan yang paling diwaspadai bagi kesehatan. "
            "Sumbernya antara lain asap kendaraan, industri, dan pembakaran.")
    with st.expander("🏭 Apa saja polutan yang lain?", expanded=False):
        st.markdown(
            "- **PM10** — debu kasar dari jalanan & konstruksi.\n"
            "- **SO₂ (sulfur dioksida)** — dari pembakaran batu bara & industri.\n"
            "- **CO (karbon monoksida)** — gas tak berbau dari knalpot kendaraan.\n"
            "- **O₃ (ozon permukaan)** — terbentuk dari reaksi gas + sinar matahari.\n"
            "- **NO₂ (nitrogen dioksida)** — dari kendaraan & industri.\n\n"
            "Standar ISPU memantau keenam polutan ini; kategori harian resmi "
            "ditentukan oleh polutan dengan indeks **tertinggi** (polutan kritis).")
    with st.expander("📊 Dari mana angka prakiraan ini?", expanded=False):
        st.markdown(
            "Angka prakiraan dihitung oleh sistem komputer yang **mempelajari pola "
            "kualitas udara dari riwayat data harian 2020–2026** — misalnya kondisi "
            "kemarin, pola musiman, dan karakter tiap titik pantau. Prakiraan bersifat "
            "perkiraan satu hari ke depan, bukan pengukuran langsung.")

with kol_e2:
    st.markdown("**Arti warna kategori — berdasarkan kadar PM2.5 "
                "(standar ISPU, Permen LHK No. 14/2020):**")
    baris_kat = ""
    for kat, w in WARNA.items():
        baris_kat += f"""
        <div style='display:flex; align-items:center; flex-wrap:wrap;
                    gap:0.45rem 0.7rem; margin-bottom:0.55rem;'>
            <span class='badge' style='background:{w["bg"]}; color:{w["teks"]}; min-width:150px; text-align:center;'>{w["muka"]} {kat}</span>
            <span style='font-size:0.85rem; color:#33475C; min-width:110px;'>{RENTANG[kat]} µg/m³</span>
            <span style='font-size:0.85rem; color:#64798D;
                  flex:1 1 13rem; min-width:13rem;'>{DESKRIPSI[kat]}</span>
        </div>"""
    st.markdown(f"<div class='kartu' style='padding:1rem 1.2rem;'>{baris_kat}</div>",
                unsafe_allow_html=True)


# ============================================================
# CATATAN & IDENTITAS
# ============================================================
st.markdown("&nbsp;")
st.markdown(f"""
<div class='catatan'>
    <strong>Catatan:</strong> Prakiraan disusun dari data simulasi periode
    2020 – Juni 2026, bukan pengukuran resmi. Kategori dan rekomendasi
    mengikuti standar ISPU (Permen LHK No. 14 Tahun 2020).
</div>
""", unsafe_allow_html=True)
