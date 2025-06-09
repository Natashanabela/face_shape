import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Load model yang akan digunakan untuk klasifikasi bentuk wajah
interpreter = tf.lite.Interpreter(model_path="model baru banget(79%).tflite")
interpreter.allocate_tensors()

# Mendapatkan informasi input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Face shape yang akan digunakan
face_shapes = ['Heart','Oblong','Oval','Round','Square']

# Inisialisasi MTCNN 
detector = MTCNN()

# CSS untuk tampilan aplikasi
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

# Inisialisasi session state 
if "captured" not in st.session_state:
    st.session_state.captured = False
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "face_shape" not in st.session_state:
    st.session_state.face_shape = None
if "upload_mode" not in st.session_state:
    st.session_state.upload_mode = False   

# Title
st.markdown("<h1 style='text-align: center;'>Let's Take a Look at Your Make-up Preference!</h1>", unsafe_allow_html=True)

# Checkbox tuntuk mengaktifkan kamera
run = st.checkbox('Run Camera', key="run_camera")
FRAME_WINDOW = st.empty()

# Mememunculkan kotak hitam jika kamera belum aktif
if not st.session_state.captured:
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    FRAME_WINDOW.image(black_frame, caption="Make sure to face forward!", channels="RGB")

if run:
    st.session_state.camera_active = True
    st.session_state.upload_mode = False
else:
    st.session_state.camera_active = False

# Fungsi untuk mendeteksi wajah dengan MTCNN dan melakuka croppiing gambar
def extract_face(image):
    results = detector.detect_faces(image)
    if not results:
        return None  

    # Mengambil bounding box yang didapatkan dari MTCNN
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    # Menyesuaikan bounding box untuk menghindari memotong fitur penting wajah 
    adj_h = 30
    new_height = height + (adj_h * 2)        
    
    adj_w = int((new_height - width) / 2) + 60
    new_width = width + (adj_w * 2)
    
    if new_width > new_height:
        new_height = new_width
    else:
        new_width = new_height
    
    new_y1 = max(y1 - adj_h, 0)
    new_y2 = min(y2 + adj_h, image.shape[0])
    new_x1 = max(x1 - adj_w, 0)
    new_x2 = min(x2 + adj_w, image.shape[1])

    new_face = image[new_y1:new_y2, new_x1:new_x2]

    # Resize menjadi 224x224
    face = cv2.resize(new_face, (224, 224))

    return np.array(face)

# Capture gambar dari kamera
if st.session_state.camera_active and not st.session_state.captured:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not access the camera.")
        st.session_state.camera_active = False

    capture_pressed = st.button("Capture")

    capturing = True
    while capturing:
        read, frame = cap.read()
        if not read:
            st.write("Error: Cannot read frame from the camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

        if capture_pressed:
            st.session_state.captured_image = frame_rgb
            st.session_state.captured = True
            st.session_state.camera_active = False
            capturing = False
            st.rerun()

    cap.release()

# Untuk mengunggah gambar dari file
if not st.session_state.captured:
    upload_file = st.file_uploader("Or Choose a File", type=["jpg", "png", "jpeg"])

    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.session_state.captured_image = image_rgb
        st.session_state.captured = True
        st.session_state.upload_mode = True
        st.rerun()

# Menampilkan gambar hasil capture
if st.session_state.captured and st.session_state.captured_image is not None:
    FRAME_WINDOW.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)

    # Button untuk retake
    if st.button("Retake"):
        st.session_state.captured = False
        st.session_state.captured_image = None
        st.session_state.face_shape = None
        st.rerun()

    # Button untuk Check bentuk wajah
    if st.button("Check"):
        face_img = extract_face(st.session_state.captured_image)

        if face_img is None:
            st.write("No face detected. Please try again.")
        else:
            # Preprocess gambar sebelum ke model
            img = face_img.astype(np.float32) / 255.0 
            img = np.expand_dims(img, axis=0)

            # Memberikan input ke model lalu melakukan prediksi
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Memnentukan kelas melalui probabilitasi paling tinggi 
            predicted_class = np.argmax(output_data)
            st.session_state.face_shape = face_shapes[predicted_class]

            # Menyimpan gambar wajah yang sudah dicrop
            buffer = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("captured_image.jpg", buffer)

            # Ke halaman check untuk melihat hasilnya
            st.switch_page("pages/Check.py")
