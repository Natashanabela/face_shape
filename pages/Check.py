import streamlit as st

# CSS untuk ngubah warna background dan hide in sidebarnya
st.markdown(
    """
    <style>
    .stApp {
        background-color: #856856;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Untuk output gambar shading dan tinting yang sesuai dengan hasilnya
face_shape_images = {
    'Heart': 'images/Heart.png', 
    'Oblong': 'images/Oblong.png',
    'Oval': 'images/Oval.png', 
    'Round': 'images/Round.png',
    'Square': 'images/Square.png' 
}

face_shape_text = {
    'Heart' : "Untuk bentuk wajah Hati, fokus untuk mempersempit atau menutupi pelipis dan tulang pipi. Lalu membuat bagian dagu supaya tidak terlalu tajam.",
    'Oblong': "Untuk bentuk wajah Oblong, fokuskan untuk membuat wajah menjadi lebih lebar dan pendek. Sehingga shadingnya pada bagian dagu dan dahi dan highligths nya pada tulang pipi.",
    'Oval': "Untuk bentuk wajah Oval, karena sudah termasuk ideal, sehingga fokus untuk mempertajam fitur wajah dan memberikan dimensi pada wajah.", 
    'Round': "Untuk bentuk wajah Bulat, fokuskan untuk membuat wajah terlihat lebih tirus dan kecil. Sehingga Shadingnya pada bagian sisi kanan dan kiri wajah dan Highlights nya pada bagian dalam wajah.",
    'Square': "Untuk bentuk wajah persegi, karena memiliki rahang yang tegas dan dahi yang lebar, fokuskan untuk memberi kesan kelembutan pada wajah dengan shading pada bagian pelipis dan rahang, lalu highlights pada tulang pipi."
}

# Mendapatkan bentuk wajah dari session state
face_shape = st.session_state.get("face_shape", "Not Detected")

# Menampilkan Gambar wajah dan hasil klasifikasi
col1, col2 = st.columns(2, gap="large")
with col1:
    st.image("captured_image.jpg", caption="Your Face")
with col2:
    st.subheader("Bentuk Wajah anda adalah,")
    st.title(face_shape)

st.header("Penempatan Shading dan Highligths untuk anda:")
# Jika bentuk wajah terdeteksi dan ada di mapping, tampilkan gambar yang sesuai
if face_shape in face_shape_images:
    st.image(face_shape_images[face_shape])
    st.subheader(face_shape_text[face_shape])
else:
    st.write("No matching face shape image found.")

# Fungsi untuk reset session state
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]  # Hapus semua session state

# Tombol untuk kembali ke halaman dashboard dan reset state
if st.button("Back"):
    reset_session()  # Reset session state sebelum pindah halaman
    st.switch_page("dashboard.py")