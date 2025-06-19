import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(layout="wide", page_title="Distribusi & Outlier Analyzer")

st.title("🔍 Aplikasi Analisis Distribusi & Outlier")
st.write("Upload data kamu dan lihat analisis distribusi serta outlier secara visual dan otomatis!")

# Upload File
uploaded_file = st.file_uploader("📤 Upload file Excel / CSV", type=["csv", "xlsx"])

if uploaded_file:
    # Baca file
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Data berhasil dimuat!")
    st.write("📊 Data Preview:")
    st.dataframe(df.head())

    # Pilih kolom numerik
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("⚠️ Tidak ada kolom numerik ditemukan.")
    else:
        col = st.selectbox("Pilih kolom numerik untuk analisis distribusi", num_cols)

        # Pilihan warna
        st.sidebar.title("🎨 Pengaturan Warna")
        main_color = st.sidebar.color_picker("Warna utama grafik", "#1f77b4")

        # Ambil data kolom terpilih
        data = df[col].dropna()
        desc = data.describe()
        q1, q3 = desc["25%"], desc["75%"]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]

        # ----------------------------------------
        # Statistik Deskriptif
        # ----------------------------------------
        st.markdown("### 📈 Statistik Deskriptif")
        st.write(desc.to_frame().T)

        # ----------------------------------------
        # Analisis Skewness (Kemiringan Distribusi)
        # ----------------------------------------
        skew_val = data.skew()
        skew_type = ""
        if skew_val > 0.5:
            skew_type = "Right Skewed (Positive Skew)"
        elif skew_val < -0.5:
            skew_type = "Left Skewed (Negative Skew)"
        else:
            skew_type = "Symmetric or Nearly Normal"

        st.markdown("### 🧭 Analisis Skewness (Kemiringan)")
        st.write(f"**Skewness Value:** {skew_val:.4f}")
        st.write(f"**Distribusi:** {skew_type}")

        # Visualisasi Skewness dengan garis mean dan median
        mean_val = data.mean()
        median_val = data.median()

        fig_skew, ax_skew = plt.subplots(figsize=(10, 5))
        sns.histplot(data, bins=30, kde=True, color=main_color, ax=ax_skew)
        ax_skew.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax_skew.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2f}')
        ax_skew.legend()
        ax_skew.set_title("Histogram dengan Garis Mean & Median")
        st.pyplot(fig_skew)

        # ----------------------------------------
        # Informasi Outlier
        # ----------------------------------------
        st.markdown("### 🚨 Informasi Outlier")
        st.write(f"Jumlah outlier: {len(outliers)}")
        st.write(outliers)

        # ----------------------------------------
        # Visualisasi Distribusi
        # ----------------------------------------
        st.markdown("### 📊 Visualisasi Distribusi")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram + KDE
        sns.histplot(data, bins=30, ax=axes[0, 0], color=main_color, kde=True)
        axes[0, 0].set_title("Histogram + KDE")

        # Boxplot
        sns.boxplot(x=data, ax=axes[0, 1], color=main_color)
        axes[0, 1].set_title("Boxplot")

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")

        # Countplot jika diskrit, KDE jika kontinu
        if data.nunique() < 20:
            sns.countplot(x=data, ax=axes[1, 1], color=main_color)
            axes[1, 1].set_title("Countplot (Diskrit)")
        else:
            sns.kdeplot(data, ax=axes[1, 1], color=main_color, fill=True)
            axes[1, 1].set_title("KDE Plot")

        st.pyplot(fig)

        # ----------------------------------------
        # Heatmap Korelasi (jika memungkinkan)
        # ----------------------------------------
        if len(num_cols) >= 2:
            st.markdown("### 🌡️ Korelasi & Heatmap Outlier (opsional)")
            corr = df[num_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

        st.info("Gunakan grafik dan statistik di atas untuk memahami distribusi, sebaran, dan outlier data kamu secara menyeluruh.")

        # ----------------------------------------
        # 📖 INTERPRETASI OTOMATIS
        # ----------------------------------------
        st.markdown("### 🧠 Interpretasi Otomatis Distribusi")

        # 1. Skewness
        if skew_val > 1:
            skew_interp = "Distribusi sangat miring ke kanan (right-skewed). Ini menunjukkan bahwa sebagian besar data berada di sisi kiri dengan beberapa nilai ekstrem di kanan."
        elif skew_val > 0.5:
            skew_interp = "Distribusi sedikit miring ke kanan. Ada indikasi beberapa nilai besar yang mempengaruhi rata-rata."
        elif skew_val < -1:
            skew_interp = "Distribusi sangat miring ke kiri (left-skewed). Ini menunjukkan bahwa sebagian besar data berada di sisi kanan dengan beberapa nilai kecil ekstrem di kiri."
        elif skew_val < -0.5:
            skew_interp = "Distribusi sedikit miring ke kiri. Ada indikasi beberapa nilai kecil yang mempengaruhi rata-rata."
        else:
            skew_interp = "Distribusi cukup simetris, mendekati distribusi normal."

        # 2. Mean vs Median
        gap = abs(mean_val - median_val)
        if gap < 0.1 * desc['std']:
            mean_median_interp = "Mean dan median sangat dekat. Ini mengindikasikan distribusi data relatif simetris tanpa bias berat."
        else:
            mean_median_interp = f"Mean ({mean_val:.2f}) dan median ({median_val:.2f}) berbeda cukup signifikan, mengindikasikan distribusi tidak simetris."

        # 3. Outlier
        prop_outlier = len(outliers) / len(data)
        if prop_outlier > 0.2:
            outlier_interp = f"Terdapat {len(outliers)} outlier ({prop_outlier:.1%} dari data). Outlier cukup banyak dan berpotensi mempengaruhi analisis statistik secara signifikan."
        elif prop_outlier > 0.05:
            outlier_interp = f"Terdapat {len(outliers)} outlier ({prop_outlier:.1%}). Outlier cukup perlu diperhatikan."
        elif len(outliers) > 0:
            outlier_interp = f"Terdapat {len(outliers)} outlier, tetapi jumlahnya relatif sedikit dan tidak terlalu mengganggu analisis."
        else:
            outlier_interp = "Tidak ditemukan outlier. Data sangat bersih."

        # 4. Variasi data
        std_val = desc['std']
        if std_val > 0.5 * desc['mean']:
            var_interp = f"Standar deviasi ({std_val:.2f}) cukup besar dibandingkan rata-rata. Ini menandakan variasi data tinggi."
        else:
            var_interp = f"Standar deviasi ({std_val:.2f}) relatif kecil terhadap rata-rata. Ini menandakan data tersebar cukup rapat."

        # Gabung dan tampilkan interpretasi
        st.markdown(f"""
        **🧾 Ringkasan Interpretasi Otomatis:**
        - {skew_interp}  
        - {mean_median_interp}  
        - {outlier_interp}  
        - {var_interp}
        """)
else:
    st.warning("📂 Silakan upload file terlebih dahulu untuk memulai analisis.")
