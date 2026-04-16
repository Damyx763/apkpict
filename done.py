import streamlit as st
import zipfile
import tempfile
import os
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Ekstrak fitur menggunakan VGG16
def extract_vgg16_features(image_path, model):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = model.predict(arr)
    return features.flatten()

# Hitung perceptual hash
def get_phash(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    return imagehash.phash(img)

# Caching model VGG16
@st.cache_resource 
def load_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return base_model

# Streamlit UI
st.title("📚 Sistem Deteksi Plagiarisme Gambar Mahasiswa")
uploaded_zip = st.file_uploader("📂 Upload file ZIP yang berisi gambar tugas mahasiswa", type=["zip"])

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        image_paths = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))

        st.success(f"✅ Total gambar ditemukan: {len(image_paths)}")

        vgg_model = load_vgg16()
        data = []
        for path in image_paths:
            features = extract_vgg16_features(path, vgg_model)
            phash = get_phash(path)
            folder_name = os.path.basename(os.path.dirname(path))
            data.append({
                'path': path,
                'filename': os.path.basename(path),
                'folder': folder_name,
                'features': features,
                'phash': phash
            })

        results = []
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                cos_sim = cosine_similarity([data[i]['features']], [data[j]['features']])[0][0]
                phash_diff = data[i]['phash'] - data[j]['phash']
                plagiarized = cos_sim > 0.9 and phash_diff <= 5
                percent_similarity = round(cos_sim * 100, 2)

                results.append({
                    'Mahasiswa 1': data[i]['folder'],
                    'File 1': data[i]['filename'],
                    'Path 1': data[i]['path'],
                    'Mahasiswa 2': data[j]['folder'],
                    'File 2': data[j]['filename'],
                    'Path 2': data[j]['path'],
                    'Cosine Similarity': round(cos_sim, 4),
                    'Persentase Kemiripan (%)': percent_similarity,
                    'pHash Difference': phash_diff,
                    'Status': "⚠️ Mirip" if plagiarized else "✅ Tidak Mirip"
                })


        df_results = pd.DataFrame(results).drop(columns=["Path 1", "Path 2"])
        st.subheader("📊 Tabel Hasil Perbandingan")
        show_only_plagiarism = st.checkbox("🔍 Tampilkan hanya yang terindikasi mirip")
        show_only_safe = st.checkbox("🟢 Tampilkan hanya yang tidak mirip (Beda)")

        if show_only_plagiarism:
            df_filtered = df_results[df_results['Status'] == "⚠️ Mirip"]
        elif show_only_safe:
            df_filtered = df_results[df_results['Status'] == "✅ Tidak Mirip"]
        else:
            df_filtered = df_results

        st.dataframe(df_filtered, use_container_width=True)
        
        # Menampilkan rata-rata dari kolom numerik
        if not df_filtered.empty:
            avg_cosine = df_filtered["Cosine Similarity"].mean()
            avg_percent = df_filtered["Persentase Kemiripan (%)"].mean()
            avg_phash = df_filtered["pHash Difference"].mean()

            st.markdown("""
            <h5 style='margin-top:20px;'><u>📈 Rata-Rata Hasil Perbandingan:</u></h5>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <ul>
                <li><b>Rata-rata Cosine Similarity:</b> {avg_cosine:.4f}</li>
                <li><b>Rata-rata Persentase Kemiripan:</b> {avg_percent:.2f}%</li>
                <li><b>Rata-rata pHash Difference:</b> {avg_phash:.2f}</li>
            </ul>
            """, unsafe_allow_html=True)



        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil sebagai CSV", data=csv, file_name="hasil_plagiarisme.csv", mime="text/csv")

       # Preview gambar sesuai filter yang dipilih 
        if (show_only_plagiarism or show_only_safe) and not df_filtered.empty:
            if show_only_plagiarism:
                st.subheader("🖼️ Preview Gambar Terindikasi Mirip")
                status_filter = "⚠️ Mirip"
            elif show_only_safe:
                st.subheader("🖼️ Preview Gambar Yang Tidak Mirip")
                status_filter = "✅ Tidak Mirip"

            total_pairs = sum(1 for r in results if r['Status'] == status_filter)
            st.markdown(f"**🔢 Total pasangan gambar yang ditampilkan: {total_pairs}**")

            for r in results:
                if r['Status'] == status_filter:
                    # Judul perbandingan
                    st.markdown(f"""
                        <h5 style="margin-bottom:10px; color: black; font-weight: bold;">
                            🧾 {r['Mahasiswa 1']} ({r['File 1']}) 🔁 {r['Mahasiswa 2']} ({r['File 2']})
                        </h5>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        img1 = Image.open(r['Path 1'])
                        st.image(img1, caption=f"{r['Mahasiswa 1']} - {r['File 1']}", use_column_width=True)
                        st.markdown(
                            f"<p style='text-align: center; font-weight: bold; color: black;'>{r['Persentase Kemiripan (%)']}% mirip</p>",
                            unsafe_allow_html=True
                        )

                    with col2:
                        img2 = Image.open(r['Path 2'])
                        st.image(img2, caption=f"{r['Mahasiswa 2']} - {r['File 2']}", use_column_width=True)
                        st.markdown(
                            f"<p style='text-align: center; font-weight: bold; color: black;'>{r['Persentase Kemiripan (%)']}% mirip</p>",
                            unsafe_allow_html=True
                        )

                    st.markdown("---")

