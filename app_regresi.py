## Aplikasi Curve Fitting Regresi Linear (Normal Equation) dan Proyeksi Ortogonal dengan Streamlit
## Gabungan curve_fitting.py dan aplikasi web dalam satu berkas
## Jalankan dengan: streamlit run app_regresi.py
import streamlit as st
import pandas as pd
import numpy as np


# Fungsi-fungsi bantu

def baca_file(file):
    """Membaca file CSV atau Excel menjadi DataFrame."""
    if file.name.lower().endswith(".csv"):
        # sep=None -> pandas mendeteksi otomatis pemisah (koma atau titik koma)
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)


def bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep):
    """Membentuk matriks desain X dan vektor target y dari DataFrame."""
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="raise")
    if data.isnull().any().any():
        raise ValueError("Terdapat sel kosong/NaN pada kolom yang dipilih.")

    X = data[kolom_fitur].to_numpy(dtype=float)
    y = data[kolom_target].to_numpy(dtype=float)

    nama_w = [f"w{i+1}" for i in range(len(kolom_fitur))]
    if pakai_intersep:
        X = np.column_stack([np.ones(len(X)), X])  # tambah kolom 1 di kiri
        nama_w = ["w0"] + nama_w
    return X, y, nama_w


def selesaikan_normal_equation(X, y):
    """Menyelesaikan (X^T X) w = X^T y dengan solver."""
    XtX = X.T @ X          # perkalian matriks
    Xty = X.T @ y
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError(
            "Matriks XᵀX singular (tidak punya invers). Kemungkinan ada fitur yang "
            "saling bergantung linear atau jumlah sampel kurang dari jumlah parameter."
        )
    w = np.linalg.solve(XtX, Xty)  # solver SPL
    return XtX, Xty, w



def hitung_evaluasi(X, y, w):
    """Mengembalikan y_hat, galat, RMSE, R2, dan jumlah galat Σ(y - y_hat)."""
    y_hat = X @ w
    galat = y - y_hat
    sse = float(np.sum(galat ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(galat ** 2)))
    jumlah_galat = round(float(np.sum(galat)), 10) + 0.0   # jumlah galat: sum(y - y_hat)
    return y_hat, galat, rmse, r2, jumlah_galat


# Halaman 1: bukti geometri (proyeksi ortogonal)

def halaman_geometri():
    st.subheader("Bukti geometri: regresi = proyeksi ortogonal")
    st.write("Data (x, y) = (1, 1), (2, 3), (3, 4). Kolom c1 = (1, 1, 1) dan c2 = (1, 2, 3) "
             "merentang sebuah bidang. P adalah proyeksi y pada bidang itu dan e = y − P.")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 4.0])
    X = np.column_stack([np.ones(len(x)), x])
    XtX, Xty, w = selesaikan_normal_equation(X, y)
    P = X @ w
    e = y - P
    c1, c2 = X[:, 0], X[:, 1]

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**XᵀX**")
        st.dataframe(pd.DataFrame(XtX, index=["w0", "w1"], columns=["w0", "w1"]))
    with k2:
        st.markdown("**Xᵀy**")
        st.dataframe(pd.DataFrame(Xty, index=["w0", "w1"], columns=["Xᵀy"]))
    st.info(f"Koefisien regresi: w0 = {w[0]:.4f}, w1 = {w[1]:.4f}")

    st.dataframe(pd.DataFrame({"c1": c1, "c2": c2, "y": y, "P = Xw": np.round(P, 4),
                               "e = y − P": np.round(e, 4)}))
    n1, n2 = st.columns(2)
    n1.metric("c1 · e", f"{round(float(c1 @ e), 10) + 0.0:.6f}")
    n2.metric("c2 · e", f"{round(float(c2 @ e), 10) + 0.0:.6f}")
    st.success("c1 · e = 0 dan c2 · e = 0, sehingga e tegak lurus pada bidang c1-c2 dan "
               "X·w sama dengan koordinat P.")


# Halaman 2: curve fitting dari file data

def halaman_regresi():
    file = st.file_uploader("Unggah file data (CSV / Excel)", type=["csv", "xlsx", "xls"])
    if file is None:
        st.info("Silakan unggah file CSV/Excel terlebih dahulu.")
        return
    try:
        df = baca_file(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    st.subheader("1. Data yang diunggah")
    st.dataframe(df)

    # ----- Pilih kolom -----
    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    if len(kolom_numerik) < 2:
        st.error("Data harus memiliki minimal 2 kolom numerik (fitur dan target).")
        return

    st.subheader("2. Pilih variabel")
    col1, col2 = st.columns(2)
    with col1:
        kolom_target = st.selectbox(
            "Kolom target (y)", kolom_numerik, index=len(kolom_numerik) - 1
        )
    with col2:
        abaikan = ("sampel", "no", "id", "nomor")
        default_fitur = [
            k for k in kolom_numerik
            if k != kolom_target and k.lower() not in abaikan
        ]
        kolom_fitur = st.multiselect(
            "Kolom fitur (x)", [k for k in kolom_numerik if k != kolom_target],
            default=default_fitur,
        )
    pakai_intersep = st.checkbox("Tambahkan intersep (w0)", value=True)

    # ----- Tombol hitung -----
    if st.button("Selesaikan", type="primary"):
        try:
            if not kolom_fitur:
                raise ValueError("Pilih minimal satu kolom fitur.")
            X, y, nama_w = bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep)
            XtX, Xty, w = selesaikan_normal_equation(X, y)
            st.session_state["hasil"] = dict(
                file=file.name, fitur=kolom_fitur, target=kolom_target,
                X=X, y=y, nama_w=nama_w, XtX=XtX, Xty=Xty, w=w,
                intersep=pakai_intersep,
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    # ----- Tampilkan hasil -----
    h = st.session_state.get("hasil")
    if not (h and h["file"] == file.name):
        return
    X, y, w, nama_w = h["X"], h["y"], h["w"], h["nama_w"]
    w = w + 0.0  # menghindari tampilan -0.0

    st.subheader("3. Langkah perhitungan")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matriks desain X**")
        st.dataframe(pd.DataFrame(X, columns=nama_w if h["intersep"] else h["fitur"]))
    with c2:
        st.markdown("**Vektor y**")
        st.dataframe(pd.DataFrame(y, columns=[h["target"]]))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**XᵀX**")
        st.dataframe(pd.DataFrame(h["XtX"], index=nama_w, columns=nama_w))
    with c4:
        st.markdown("**Xᵀy**")
        st.dataframe(pd.DataFrame(h["Xty"], index=nama_w, columns=["Xᵀy"]))

    st.subheader("4. Solusi w")
    hasil_w = pd.DataFrame({"parameter": nama_w, "nilai": np.round(w, 4)})
    st.dataframe(hasil_w, hide_index=True)

    # Persamaan model
    suku = []
    for nama, nilai, fitur in zip(nama_w, w, ([None] + h["fitur"]) if h["intersep"] else h["fitur"]):
        suku.append(f"{nilai:.4f}" if fitur is None else f"{nilai:.4f}·{fitur}")
    st.info(f"Model: {h['target']} = " + " + ".join(suku))

    # Evaluasi
    y_hat, galat, rmse, r2, jumlah_galat = hitung_evaluasi(X, y, w)
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:.4f}")
    m2.metric("R²", f"{r2:.4f}")
    m3.metric("Jumlah galat Σ(y − ŷ)", f"{jumlah_galat:.6f}")
    st.caption("Jika model memakai intersep, kolom 1 pada X ortogonal terhadap galat "
               "(1ᵀe = Σe = 0), sehingga jumlah galat mendekati 0.")

    st.markdown("**Perbandingan nilai aktual dan prediksi**")
    df_pred = pd.DataFrame({"aktual": y, "prediksi": np.round(y_hat, 4),
                            "galat": np.round(galat, 4)})
    st.dataframe(df_pred)

    st.download_button(
        "Unduh hasil prediksi (CSV)",
        df_pred.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi.csv", mime="text/csv",
    )

    # Prediksi data baru
    st.subheader("5. Prediksi data baru")
    nilai_baru = []
    kolom_in = st.columns(len(h["fitur"]))
    for kol, nama in zip(kolom_in, h["fitur"]):
        nilai_baru.append(kol.number_input(nama, value=float(df[nama].mean())))
    x_baru = np.array(([1.0] if h["intersep"] else []) + nilai_baru)
    st.success(f"Prediksi {h['target']} = {float(x_baru @ w):.2f}")


# Tampilan utama

st.set_page_config(page_title="Curve Fitting Regresi Linear", layout="wide")
st.title("Curve Fitting Regresi Linear dan Proyeksi Ortogonal")
st.caption("Regresi linear berganda melalui normal equation: (XᵀX) w = Xᵀy")

tab_regresi, tab_geometri = st.tabs(["Regresi dari data", "Bukti geometri"])
with tab_geometri:
    halaman_geometri()
with tab_regresi:
    halaman_regresi()
