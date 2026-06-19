import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import numpy as np
import time
from collections import Counter
from pathlib import Path
import plotly.express as px

# ============================================================
# Configuration page
# ============================================================

st.set_page_config(
    page_title="Traffic Surveillance AI",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# Style moderne
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .stApp {
        background-color: #0f172a;
        color: white;
    }

    h1, h2, h3 {
        color: white;
    }

    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
        /* =========================
    Boutons modernes
    ========================= */

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }

    /* Hover */
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(59,130,246,0.5);
        color: white;
    }

    /* Click */
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Focus */
    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.4);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Titre principal
# ============================================================

st.title("🚦 Traffic Surveillance System")
st.markdown("### YOLO11 + ByteTrack + UA-DETRAC")

# ============================================================
# Chargement modèle
# ============================================================

MODEL_PATH = "models/best.pt"

@st.cache_resource

def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# Classes
# ============================================================

CLASS_NAMES = {
    0: "others_truck",
    1: "car",
    2: "van",
    3: "bus"
}

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Paramètres")

confidence = st.sidebar.slider(
    "Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

source_option = st.sidebar.radio(
    "Source vidéo",
    ["Upload vidéo", "Webcam"]
)

show_tracking = st.sidebar.checkbox(
    "Activer ByteTrack",
    value=True
)

if "stop_webcam" not in st.session_state:
    st.session_state.stop_webcam = False

# ============================================================
# Statistiques
# ============================================================

stats_placeholder = st.empty()

# ============================================================
# Colonnes principales
# ============================================================

col1, col2 = st.columns([3, 1])

# ============================================================
# Upload vidéo
# ============================================================

video_source = None

if source_option == "Upload vidéo":

    uploaded_video = st.file_uploader(
        "📁 Charger une vidéo",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        video_source = temp_video.name

# ============================================================
# Webcam
# ============================================================

elif source_option == "Webcam":
    video_source = 0

    if st.sidebar.button("⏹️ Arrêter la webcam", key="stop_webcam_button"):
        st.session_state.stop_webcam = True

# ============================================================
# Lancement analyse
# ============================================================

if st.button("▶️ Démarrer l'analyse"):
    st.session_state.stop_webcam = False

    if video_source is None:
        st.warning("Veuillez charger une vidéo")

    else:

        cap = cv2.VideoCapture(video_source)

        frame_placeholder = st.empty()
        chart_placeholder = st.empty()
        total_counter = Counter()

        fps_list = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            start_time = time.time()

            if st.session_state.stop_webcam:
                break

            # ====================================================
            # Tracking ByteTrack
            # ====================================================

            if show_tracking:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    verbose=False
                )
            else:
                results = model.predict(
                    frame,
                    conf=confidence,
                    verbose=False
                )

            annotated_frame = results[0].plot()

            # ====================================================
            # Comptage objets
            # ====================================================

            current_counter = Counter()

            if results[0].boxes is not None:

                classes = results[0].boxes.cls.cpu().numpy()

                for cls in classes:
                    class_name = CLASS_NAMES.get(int(cls), "unknown")
                    current_counter[class_name] += 1
                    total_counter[class_name] += 1

            # ====================================================
            # FPS
            # ====================================================

            fps = 1 / (time.time() - start_time)
            fps_list.append(fps)

            avg_fps = np.mean(fps_list)

            # ====================================================
            # RGB conversion
            # ====================================================

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

            # ====================================================
            # Dashboard stats
            # ====================================================

            with stats_placeholder.container():

                c1, c2 = st.columns(2)

                with c1:
                    st.metric("🚗 Cars", current_counter["car"])
                    st.metric("🚌 Bus", current_counter["bus"])

                with c2:
                    st.metric("🚐 Vans", current_counter["van"])
                    st.metric("⚡ FPS", round(avg_fps, 2))

            # ====================================================
            # Graphique live
            # ====================================================

            df = pd.DataFrame({
                "Classe": list(total_counter.keys()),
                "Count": list(total_counter.values())
            })

            if len(df) > 0:
                fig = px.bar(
                    df,
                    x="Classe",
                    y="Count",
                    title="Détections cumulées"
                )

                chart_placeholder.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"chart_{time.time()}"
                )

        cap.release()

        st.success("Analyse terminée")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    "### Projet Deep Learning — Traffic Surveillance with YOLO11 & ByteTrack"
)
