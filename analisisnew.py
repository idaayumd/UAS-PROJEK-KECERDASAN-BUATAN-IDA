import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

# --- Configuration --- #
RGB_PALETTE = ['#66545e', '#a39193', '#aa6f73', '#eea990', '#f6e0b5']
sns.set_palette(RGB_PALETTE)

# --- Helper Functions (Replicating Notebook Steps) ---

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Make a copy to avoid modifying the original DataFrame passed
    df_copy = df.copy()

    # 1. Date Feature Engineering
    try:
        df_copy['defect_date'] = pd.to_datetime(df_copy['defect_date'], format='%m/%d/%Y')
        df_copy['defect_month'] = df_copy['defect_date'].dt.month
        df_copy['defect_day_of_week'] = df_copy['defect_date'].dt.dayofweek
        df_copy['defect_day'] = df_copy['defect_date'].dt.day
        df_copy['defect_year'] = df_copy['defect_date'].dt.year
        df_copy = df_copy.drop('defect_date', axis=1)
    except Exception as e:
        st.warning(f"Error converting 'defect_date': {e}. Skipping date feature engineering.")

    # 2. Categorical Encoding
    categorical_cols = df_copy.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)

    # 3. Dropping Irrelevant Columns
    columns_to_drop = []
    if 'defect_id' in df_encoded.columns:
        columns_to_drop.append('defect_id')
    if 'product_id' in df_encoded.columns:
        columns_to_drop.append('product_id')

    df_processed = df_encoded.drop(columns=columns_to_drop, errors='ignore')

    # 4. Outlier Handling (Simplified for Streamlit - only capping for numericals)
    numerical_cols_for_outliers = df_processed.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols_for_outliers:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
        df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])

    # 5. Data Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns, index=df_processed.index)

    return df_scaled, df_processed # Return both scaled and processed (pre-scaled) for analysis

def run_kmeans(df_scaled, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

def plot_pca_clusters(df_scaled, clusters, RGB_PALETTE):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cluster',
        palette=RGB_PALETTE,
        data=pca_df,
        legend='full',
        alpha=0.7,
        ax=ax
    )
    ax.set_title('Clusters Visualized with PCA (2 Components)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    return pca.explained_variance_ratio_.sum()


# --- Streamlit App Layout --- #
st.set_page_config(layout="wide", page_title="Defect Clustering Analysis")

st.title("Analisis Clustering Cacat Produk")
st.markdown("### Oleh: Ida Ayu M.D")
st.write("Aplikasi ini melakukan analisis clustering pada dataset cacat produk, dengan preprocessing, K-Means, dan interpretasi fitur.")

import streamlit as st
# CSS background dengan pattern bunga
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #ffe6eb, #ffffff), url("D:\KuliahnewD\projekAI\lily.png");
            background-size: 150px;   /* ukuran motif bunga */
            background-repeat: repeat; /* jadi pattern */
            background-attachment: fixed;
            background-position: center; /* titik awal pengulangan */
        }
        h1, h3, p, label {
            position: relative;
            z-index: 10;
            text-shadow: 1px 1px 2px #ffffff; /* biar teks tetap jelas */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- File Upload --- #
st.header("1. Unggah Dataset Anda")
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("Dataset berhasil dimuat!")
    st.subheader("Preview Data:")
    st.dataframe(df.head())

    st.header("2. Pra-pemrosesan Data")
    st.info("Melakukan rekayasa fitur tanggal, one-hot encoding, penanganan outlier, dan penskalaan data.")

    # Cache preprocessed data to avoid re-running on every interaction
    df_scaled, df_processed_original_features = preprocess_data(df)

    st.success("Data berhasil dipra-proses dan diskalakan.")
    st.subheader("Preview Data Skala:")
    st.dataframe(df_scaled.head())

    st.header("3. Clustering K-Means")
    optimal_k = st.slider("Pilih Jumlah Cluster (K):", min_value=2, max_value=10, value=3)
    st.write(f"Menggunakan K = {optimal_k} untuk K-Means Clustering.")

    clusters, kmeans_model = run_kmeans(df_scaled, optimal_k)
    df_processed_original_features['Cluster'] = clusters

    st.subheader("Distribusi Cluster:")
    st.write(df_processed_original_features['Cluster'].value_counts())

    st.subheader("Visualisasi Cluster (PCA):")
    total_explained_variance = plot_pca_clusters(df_scaled, clusters, RGB_PALETTE)
    st.write(f"Varians total yang dijelaskan oleh 2 PC: {total_explained_variance:.2f}%")

    st.subheader("Profil Rata-rata Fitur per Cluster:")
    cluster_means = df_processed_original_features.groupby('Cluster').mean()
    st.dataframe(cluster_means)

  st.subheader("Interpretasi Cluster")
    for i in range(optimal_k):
        st.write(f"#### Cluster {i}:")
        # Select top 3 features with highest absolute mean values in this cluster
        relevant_features = cluster_means.loc[i][cluster_means.loc[i].abs() > 0.05] # Filter out near-zero means
        top_features = relevant_features.nlargest(3).index.tolist()
        bottom_features = relevant_features.nsmallest(3).index.tolist()

        if top_features:
            st.write(f"  - Karakteristik utama (nilai tinggi): {', '.join(top_features)}")
        if bottom_features:
            st.write(f"  - Karakteristik utama (nilai rendah): {', '.join(bottom_features)}")

        st.markdown("---")  # garis pemisah visual
        # Tambahkan interpretasi bisnis otomatis per cluster
        if i == 0:
           st.markdown("→ Produk dengan cacat ringan, biaya perbaikan relatif rendah. Cocok diperbaiki di tahap inspeksi akhir.")
        elif i == 1:
           st.markdown("→ Produk dengan cacat sedang, biaya perbaikan menengah. Perlu evaluasi proses produksi agar cacat tidak berulang.")
        elif i == 2:
           st.markdown("→ Produk dengan cacat berat, biaya perbaikan tinggi. Indikasi masalah serius di lini produksi.")
        else:
           st.markdown("→ Cluster tambahan: analisis sesuai karakteristik fitur dominan.")
else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
