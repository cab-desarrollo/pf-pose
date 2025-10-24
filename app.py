# -----------------------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
import io
import cv2 # OpenCV para procesamiento y dibujo
import mediapipe as mp # Para pose estimation
import numpy as np
import math # Para cálculos trigonométricos si son necesarios

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA PÁGINA Y CONSTANTES
# -----------------------------------------------------------------------------

# --- Constantes de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo", "cab-logo.png")
POSE_IMG_DIR = os.path.join(ASSETS_DIR, "pose_img")
USERS_CSV_PATH = os.path.join(BASE_DIR, "users.csv")

st.set_page_config(
    page_title="CAB - Análisis Biomecánico Postural",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Nombres Descriptivos para las Poses ---
POSE_FILES_INFO = {
    "1_static_front_view": {"title": "1. Vista Anterior (Estática)", "index": 1},
    "2_static_lateral_view": {"title": "2. Vista Sagital (Estática)", "index": 2},
    "3_static_posterior_view": {"title": "3. Vista Posterior (Estática)", "index": 3},
    "4_squat_front_view": {"title": "4. Sentadilla OHS (Frontal)", "index": 4},
    "5_squat_lateral_view": {"title": "5. Sentadilla OHS (Sagital)", "index": 5},
    "6_squat_posterior_view": {"title": "6. Sentadilla OHS (Posterior)", "index": 6},
    "7_sls_left_view": {"title": "7. Sentadilla Unipodal (Izq)", "index": 7},
    "8_sls_right_view": {"title": "8. Sentadilla Unipodal (Der)", "index": 8}
}

# --- Configuración MediaPipe ---
mp_pose = mp.solutions.pose

# The model path is usually predictable inside the installed package
MODEL_FILE_NAME = "pose_landmark_heavy.tflite"
MP_SOLUTION_PATH = os.path.dirname(mp_pose.__file__)

# Try common model paths (adjust '2' for model_complexity)
MODEL_PATH = os.path.join(MP_SOLUTION_PATH, 'modules', 'pose_landmark', MODEL_FILE_NAME)

# Use st.cache_resource to load the model file's bytes once.
@st.cache_resource
def get_pose_model_path():
    """Returns the absolute path to the heavy pose model file."""
    # Check expected installation location
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    else:
        # Fallback check (less common, but safe)
        fallback_path = os.path.join(os.path.dirname(mp.__file__), 'modules', 'pose_landmark', MODEL_FILE_NAME)
        if os.path.exists(fallback_path):
            return fallback_path

        # If the model is STILL not found, it means the installation failed to include it.
        # This is unlikely after successful pip install, but worth reporting.
        raise FileNotFoundError(f"MediaPipe Pose Model not found at expected location: {MODEL_PATH}")


# The actual Pose object MUST be created by passing the model_asset_path argument,
# which bypasses the automatic download logic.
@st.cache_resource
def initialize_pose_detector():
    """Inicializa y cachea el detector de Pose de MediaPipe."""
    # Al estar MEDIAPIPE_ENABLE_DOWNLOADS="false", este constructor funcionará.
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

pose_detector = initialize_pose_detector()
mp_drawing = mp.solutions.drawing_utils

# --- Estilos de Dibujo ---
COLOR_ESQUELETO = (230, 230, 230)
COLOR_PUNTO = (0, 255, 0)
GROSOR_LINEA = 1
RADIO_PUNTO = 3

st.markdown("""
<style>
    /* --- Estilos Contenedor Pose (Menú Principal) --- */
    .pose-container {
        text-align: center;
        margin-bottom: 1rem;
        overflow: hidden;
        border-radius: 8px;
        background-color: #f0f2f6;
        padding-bottom: 5px;
    }
    .pose-container [data-testid="stButton"] > button {
        display: block; width: 100% !important;
        background-color: #FFFFFF !important; color: lightblue !important;
        border: 2px solid lightblue !important; font-weight: bold !important;
        font-size: 1.1rem !important; text-align: center !important;
        padding: 8px 0 !important; border-radius: 0 !important;
        border-top-left-radius: 8px !important; border-top-right-radius: 8px !important;
        margin: 0 !important;
        transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
    }
    .pose-container [data-testid="stButton"] > button:hover {
        background-color: #000030 !important; color: white !important;
        border-color: #000030 !important;
    }
    .pose-container [data-testid="stButton"] > button:focus {
        background-color: #000030 !important; color: white !important;
        border-color: #000030 !important; box-shadow: none !important;
    }

    /* --- Estilos Imagen Pose (Menú Principal) --- */
    .pose-container .stImage { padding-left: 8px; padding-right: 8px; }
    .pose-container .stImage img {
        display: block; width: 100%; height: auto;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        cursor: pointer; border-radius: 0;
        border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
        margin-top: 0;
    }
    .pose-container .stImage img:hover {
        transform: scale(1.03); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
    }

    /* --- Estilos Botones Header (Vista Detalle) --- */
    /* Apuntar a los botones dentro del bloque horizontal superior */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button {
        margin-bottom: 0px !important; /* NUEVO: Eliminar margen inferior excesivo */
    }

    /* Estilo Botón Secundario (Cerrar Sesión) en dark mode */
     div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button[kind="secondary"] {
        color: #FAFAFA !important; border: 1px solid #FAFAFA !important;
    }
     div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button[kind="secondary"]:hover {
        color: lightblue !important; border-color: lightblue !important;
        background-color: rgba(0, 0, 80, 0.1) !important;
    }
     div[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button[kind="secondary"]:focus {
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. FUNCIONES DE AUTENTICACIÓN Y DATOS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_users(csv_path):
    # (Misma función que antes)
    if not os.path.exists(csv_path): st.error(f"Error Crítico: No se encontró '{os.path.basename(csv_path)}'."); return None
    try:
        users_df = pd.read_csv(csv_path)
        if 'username' not in users_df.columns or 'password' not in users_df.columns: st.error("Error Crítico: 'users.csv' debe tener 'username' y 'password'."); return None
        users_df['username'] = users_df['username'].astype(str); users_df['password'] = users_df['password'].astype(str)
        return users_df
    except pd.errors.EmptyDataError: st.error("Error Crítico: 'users.csv' está vacío."); return None
    except Exception as e: st.error(f"Error Crítico al leer 'users.csv': {e}"); return None

def check_login(username, password, users_df):
    # (Misma función que antes)
    if users_df is None: return False
    if username is None or password is None: return False
    user_record = users_df[users_df['username'] == str(username)]
    if not user_record.empty:
        if user_record.iloc[0]['password'] == str(password): return True
    return False

# -----------------------------------------------------------------------------
# 4. FUNCIONES DE PROCESAMIENTO Y ANÁLISIS
# -----------------------------------------------------------------------------

def calcular_angulo_3p(a, b, c):
    """Calcula el ángulo (grados) a-b-c (vértice en b). Coords (x, y)"""
    if a is None or b is None or c is None: return None
    try:
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return None
        cos_angle = dot / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except Exception as e:
        print(f"Error calculando ángulo 3p: {e}"); return None

def calcular_angulo_linea_horizontal(p1, p2):
    """Calcula ángulo (grados) de la línea p1-p2 con la horizontal."""
    if p1 is None or p2 is None: return None
    try:
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        if delta_x == 0: return 90.0 if delta_y != 0 else 0.0
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except Exception as e:
        print(f"Error calculando ángulo con horizontal: {e}"); return None

def calcular_angulo_linea_vertical(p1, p2):
    """Calcula ángulo (grados) de la línea p1-p2 con la vertical (+Y hacia abajo)."""
    if p1 is None or p2 is None: return None
    try:
        delta_y = p2[1] - p1[1] # Y es positivo hacia abajo
        delta_x = p2[0] - p1[0]
        if delta_y == 0: return 90.0 if delta_x != 0 else 0.0
        angle_rad = np.arctan2(delta_x, delta_y) # Invertido para ángulo con Y
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except Exception as e:
        print(f"Error calculando ángulo con vertical: {e}"); return None

def obtener_coords(landmarks, landmark_index, img_width, img_height):
    """Obtiene coords (x, y) en píxeles si el landmark es visible y está presente."""
    try:
        idx_value = landmark_index.value if hasattr(landmark_index, 'value') else landmark_index
        if idx_value < 0 or idx_value >= len(landmarks):
            return None
        lm = landmarks[idx_value]
        if not all(hasattr(lm, attr) for attr in ['x', 'y', 'visibility']):
            return None
        if lm.visibility > 0.5:
             if 0 <= lm.x <= 1.5 and 0 <= lm.y <= 1.5:
                 return (int(lm.x * img_width), int(lm.y * img_height))
             else:
                 return None
        else:
            return None
    except Exception as e:
        print(f"Excepción obteniendo coords para índice {landmark_index}: {e}")
        return None


def generar_analisis_texto(analysis_angles, pose_index):
    """
    Genera el texto formateado para el archivo .txt
    usando la base de conocimiento de 'get_explanation_for_pose'.
    """
    pose_info = get_explanation_for_pose(pose_index)
    pose_title = POSE_FILES_INFO.get(f"{pose_index}_...".split('_', 1)[1], {}).get('title', f"Pose {pose_index}")

    lines = [f"Análisis Biomecánico - {pose_title}\n", "="*30 + "\n"]

    if not analysis_angles:
        lines.append("No se pudieron calcular ángulos relevantes (baja visibilidad de puntos clave?).")
        return "".join(lines)

    lines.append(f"Objetivo de la Evaluación:\n{pose_info.get('significado_biomecanico', 'N/A')}\n\n")
    lines.append("--- Análisis Cuantitativo ---\n")

    for metrica in pose_info.get("metricas_clave", []):
        nombre = metrica["nombre"]
        clave = metrica["clave_resultado"]
        valor = analysis_angles.get(clave)

        if valor is not None:
            norma_min = metrica["norma_min"]
            norma_max = metrica["norma_max"]
            lines.append(f"- {nombre}: {valor:.1f}°   (Norma: {norma_min}° a {norma_max}°)\n")
            lines.append(f"  Interpretación: {metrica['interpretacion']}\n")
        else:
            lines.append(f"- {nombre}: No calculado\n")

    lines.append("\n--- Métricas Secundarias ---\n")
    for metrica in pose_info.get("metricas_secundarias", []):
        nombre = metrica["nombre"]
        clave = metrica["clave_resultado"]
        valor = analysis_angles.get(clave)

        if valor is not None:
            lines.append(f"- {nombre}: {valor:.1f}°\n")
        else:
            lines.append(f"- {nombre}: No calculado\n")

    return "".join(lines)


def process_uploaded_image(uploaded_file_bytes, pose_index):
    """
    Procesa imagen, calcula métricas avanzadas del informe y
    devuelve bytes de imagen, texto de análisis y un dict con los ángulos.
    (Esta función no requiere cambios, la lógica de cálculo es la misma)
    """
    analysis_angles = {}
    skeleton_image_bytes = None
    analysis_text = f"Análisis para Pose {pose_index}\n(Procesamiento no completado)"

    try:
        # 1. Decodificar y Preparar Imagen
        file_bytes = np.asarray(bytearray(uploaded_file_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None: return None, "Error: No se pudo decodificar la imagen.", {}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_rgb.shape
        img_rgb.flags.writeable = False

        # 2. Detectar Pose
        results = pose_detector.process(img_rgb)

        # 3. Preparar Imagen para Dibujar (copia BGR)
        img_to_draw = img_bgr.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            lm_idx = mp_pose.PoseLandmark

            # 4. Dibujar Esqueleto Básico
            mp_drawing.draw_landmarks(
                img_to_draw,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_PUNTO, thickness=-1, circle_radius=RADIO_PUNTO),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_ESQUELETO, thickness=GROSOR_LINEA)
            )

            # 5. Obtener Coordenadas
            coords = {lm_name: obtener_coords(landmarks, lm_enum, img_w, img_h)
                      for lm_name, lm_enum in lm_idx.__members__.items()}

            # Calcular puntos virtuales (C7 y Centro Pélvico)
            if coords["LEFT_SHOULDER"] and coords["RIGHT_SHOULDER"]:
                coords["MID_SHOULDER"] = ( (coords["LEFT_SHOULDER"][0] + coords["RIGHT_SHOULDER"][0]) / 2,
                                           (coords["LEFT_SHOULDER"][1] + coords["RIGHT_SHOULDER"][1]) / 2 )
            else: coords["MID_SHOULDER"] = None

            if coords["LEFT_HIP"] and coords["RIGHT_HIP"]:
                coords["MID_HIP"] = ( (coords["LEFT_HIP"][0] + coords["RIGHT_HIP"][0]) / 2,
                                      (coords["LEFT_HIP"][1] + coords["RIGHT_HIP"][1]) / 2 )
            else: coords["MID_HIP"] = None

            if coords["LEFT_ANKLE"] and coords["RIGHT_ANKLE"]:
                coords["MID_ANKLE"] = ( (coords["LEFT_ANKLE"][0] + coords["RIGHT_ANKLE"][0]) / 2,
                                      (coords["LEFT_ANKLE"][1] + coords["RIGHT_ANKLE"][1]) / 2 )
            else: coords["MID_ANKLE"] = None

            # Detección automática de lado para vistas sagitales
            lado_visible = "RIGHT"
            if coords["LEFT_SHOULDER"] and coords["RIGHT_SHOULDER"]:
                if landmarks[lm_idx.LEFT_SHOULDER.value].visibility > landmarks[lm_idx.RIGHT_SHOULDER.value].visibility:
                    lado_visible = "LEFT"
            elif coords["LEFT_SHOULDER"]:
                 lado_visible = "LEFT"


            # 6. Calcular Ángulos Específicos por Pose (Métricas del Informe)

            if pose_index == 1: # Vista Anterior Estática
                analysis_angles["angulo_cabeza_h"] = calcular_angulo_linea_horizontal(coords["LEFT_EYE"], coords["RIGHT_EYE"]) # NUEVO
                analysis_angles["angulo_hombros_h"] = calcular_angulo_linea_horizontal(coords["LEFT_SHOULDER"], coords["RIGHT_SHOULDER"])
                analysis_angles["angulo_pelvis_h"] = calcular_angulo_linea_horizontal(coords["LEFT_HIP"], coords["RIGHT_HIP"])
                analysis_angles["angulo_FPPA_estatico_der"] = calcular_angulo_3p(coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"])
                analysis_angles["angulo_FPPA_estatico_izq"] = calcular_angulo_3p(coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"])

            elif pose_index == 2: # Vista Sagital Estática
                l_ear, l_shoulder, l_hip, l_knee, l_ankle = f"{lado_visible}_EAR", f"{lado_visible}_SHOULDER", f"{lado_visible}_HIP", f"{lado_visible}_KNEE", f"{lado_visible}_ANKLE"
                if coords["MID_SHOULDER"] and coords[l_ear]:
                    punto_horizontal = (coords["MID_SHOULDER"][0] + 100, coords["MID_SHOULDER"][1])
                    analysis_angles["angulo_CVA"] = calcular_angulo_3p(punto_horizontal, coords["MID_SHOULDER"], coords[l_ear]) # NUEVO

                analysis_angles["inclinacion_corporal_v"] = calcular_angulo_linea_vertical(coords[l_ankle], coords[l_shoulder]) # NUEVO
                analysis_angles["inclinacion_pelvica_v"] = calcular_angulo_linea_vertical(coords[l_hip], coords[l_shoulder])
                analysis_angles["extension_rodilla"] = calcular_angulo_3p(coords[l_hip], coords[l_knee], coords[l_ankle])

            elif pose_index == 3: # Vista Posterior Estática
                analysis_angles["angulo_hombros_h"] = calcular_angulo_linea_horizontal(coords["LEFT_SHOULDER"], coords["RIGHT_SHOULDER"])
                analysis_angles["angulo_pelvis_h"] = calcular_angulo_linea_horizontal(coords["LEFT_HIP"], coords["RIGHT_HIP"])
                analysis_angles["angulo_talon_r_v"] = calcular_angulo_linea_vertical(coords["RIGHT_ANKLE"], coords["RIGHT_HEEL"])
                analysis_angles["angulo_talon_l_v"] = calcular_angulo_linea_vertical(coords["LEFT_ANKLE"], coords["LEFT_HEEL"])

            elif pose_index == 4: # OHS Frontal
                analysis_angles["angulo_pelvis_h"] = calcular_angulo_linea_horizontal(coords["LEFT_HIP"], coords["RIGHT_HIP"])
                analysis_angles["angulo_FPPA_dinamico_der"] = calcular_angulo_3p(coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"])
                analysis_angles["angulo_FPPA_dinamico_izq"] = calcular_angulo_3p(coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"])
                if coords["MID_HIP"] and coords["MID_ANKLE"]:
                    analysis_angles["desplazamiento_pelvico_px"] = coords["MID_HIP"][0] - coords["MID_ANKLE"][0]

            elif pose_index == 5: # OHS Sagital
                l_shoulder, l_hip, l_knee, l_ankle, l_foot = f"{lado_visible}_SHOULDER", f"{lado_visible}_HIP", f"{lado_visible}_KNEE", f"{lado_visible}_ANKLE", f"{lado_visible}_FOOT_INDEX"
                analysis_angles["inclinacion_tronco_v"] = calcular_angulo_linea_vertical(coords[l_hip], coords[l_shoulder])
                analysis_angles["inclinacion_tibia_v"] = calcular_angulo_linea_vertical(coords[l_ankle], coords[l_knee]) # NUEVO
                if analysis_angles.get("inclinacion_tronco_v") is not None and analysis_angles.get("inclinacion_tibia_v") is not None:
                    analysis_angles["paralelismo_tronco_tibia"] = analysis_angles["inclinacion_tronco_v"] - analysis_angles["inclinacion_tibia_v"] # NUEVO

                analysis_angles["flexion_cadera"] = calcular_angulo_3p(coords[l_shoulder], coords[l_hip], coords[l_knee])
                analysis_angles["flexion_rodilla"] = calcular_angulo_3p(coords[l_hip], coords[l_knee], coords[l_ankle])
                analysis_angles["dorsiflexion_tobillo"] = calcular_angulo_3p(coords[l_knee], coords[l_ankle], coords[l_foot])

            elif pose_index == 6: # OHS Posterior
                analysis_angles["angulo_talon_r_v"] = calcular_angulo_linea_vertical(coords["RIGHT_ANKLE"], coords["RIGHT_HEEL"])
                analysis_angles["angulo_talon_l_v"] = calcular_angulo_linea_vertical(coords["LEFT_ANKLE"], coords["LEFT_HEEL"])

            elif pose_index == 7: # SLS Izquierda
                analysis_angles["angulo_CPD"] = calcular_angulo_linea_horizontal(coords["LEFT_HIP"], coords["RIGHT_HIP"])
                analysis_angles["angulo_FPPA_sls_izq"] = calcular_angulo_3p(coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"])
                analysis_angles["inclinacion_tronco_v"] = calcular_angulo_linea_vertical(coords["MID_HIP"], coords["MID_SHOULDER"]) # NUEVO
                analysis_angles["flexion_rodilla_sls_izq"] = calcular_angulo_3p(coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"])

            elif pose_index == 8: # SLS Derecha
                analysis_angles["angulo_CPD"] = calcular_angulo_linea_horizontal(coords["LEFT_HIP"], coords["RIGHT_HIP"])
                analysis_angles["angulo_FPPA_sls_der"] = calcular_angulo_3p(coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"])
                analysis_angles["inclinacion_tronco_v"] = calcular_angulo_linea_vertical(coords["MID_HIP"], coords["MID_SHOULDER"]) # NUEVO
                analysis_angles["flexion_rodilla_sls_der"] = calcular_angulo_3p(coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"])

            # 7. Generar Texto de Análisis
            analysis_text = generar_analisis_texto(analysis_angles, pose_index)

        else:
            analysis_text = f"Análisis para Pose {pose_index}\nAdvertencia: No se detectó pose en la imagen."
            img_to_draw = img_bgr

        # 8. Codificar Imagen (esqueleto) a Bytes
        is_success, buffer = cv2.imencode(".png", img_to_draw)
        if is_success:
            skeleton_image_bytes = io.BytesIO(buffer).getvalue()
        else:
            analysis_text += "\nError: No se pudo codificar la imagen del esqueleto."

        return skeleton_image_bytes, analysis_text, analysis_angles

    except Exception as e:
        return None, f"Error durante el procesamiento general: {e}", {}

# --- Función auxiliar para obtener explicaciones ---
def get_explanation_for_pose(pose_index):
    """
    Base de conocimiento actualizada con descripciones completas del protocolo
    y métricas ajustadas según el último informe.
    """
    explanations = {
        1: { # VISTA ANTERIOR ESTÁTICA
            "significado_biomecanico": "Establecer la línea base estructural del atleta, evaluando la simetría corporal horizontal y la alineación de las extremidades inferiores en bipedestación neutra.",
            "descripcion_completa": """
                **Nombre Técnico:** Evaluación Postural Estática – Plano Frontal.
                **Key Landmarks:** Hombros (11, 12), Caderas (23, 24), Rodillas (25, 26), Tobillos (27, 28).
                **Análisis:**
                - **Nivelación Pélvica:** Ángulo (23-24) vs. Horizontal. Una desviación > 2° puede indicar oblicuidad pélvica, dismetría.
                - **Alineación Rodillas:** Ángulo (Cadera-Rodilla-Tobillo) (ej. 24-26-28). Buscar alineación neutra (~180°). Desviaciones: Genu Valgo (<175°), Genu Varo (>185°).
                - **Arco Plantar:** Observación cualitativa. Colapso (pie plano) afecta absorción/propulsión.
                **Alertas:** Asimetría pélvica/hombros, valgo/varo pronunciado, pronación excesiva.
            """,
            "metricas_clave": [
                {"nombre": "Nivelación Pélvica", "clave_resultado": "angulo_pelvis_h", "norma_min": -2.0, "norma_max": 2.0, "interpretacion": "Oblicuidad pélvica (>2°)."},
                {"nombre": "Alineación Rodilla (Der)", "clave_resultado": "angulo_FPPA_estatico_der", "norma_min": 175.0, "norma_max": 185.0, "interpretacion": "Valgo (<175°) / Varo (>185°) derecho."},
                {"nombre": "Alineación Rodilla (Izq)", "clave_resultado": "angulo_FPPA_estatico_izq", "norma_min": 175.0, "norma_max": 185.0, "interpretacion": "Valgo (<175°) / Varo (>185°) izquierdo."}
            ],
            "metricas_secundarias": [
                {"nombre": "Nivel de Hombros", "clave_resultado": "angulo_hombros_h", "norma_min": -2.0, "norma_max": 2.0, "interpretacion": "Asimetría de hombros."}
            ]
        },
        2: { # VISTA SAGITAL ESTÁTICA
            "significado_biomecanico": "Evaluar las curvaturas fisiológicas de la columna vertebral y la alineación vertical de las articulaciones clave respecto a la línea de gravedad.",
             "descripcion_completa": """
                **Nombre Técnico:** Evaluación Postural Estática – Plano Sagital (Línea de Plomada).
                **Key Landmarks:** Oreja (8), Hombro (12), Cadera (24), Rodilla (26), Tobillo (28) (lado visible).
                **Análisis:**
                - **Alineación Vertical:** Evaluar posición horizontal (X) de 8, 12, 24, 26 relativa a 28. Ideal: línea vertical.
                - **Inclinación Pélvica:** Ángulo (Hombro-Cadera) (12-24) vs. Vertical. Anteversión se asocia a hiperlordosis; retroversión a rectificación lumbar.
                - **Posición Cefálica (CVA):** Ángulo Craniovertebral (calculado con C7 virtual). Norma: 47-50°. Bajo indica FHP (Cabeza adelantada).
                - **Posición Hombros:** Observar Hombro (12) adelantado respecto a Cadera (24) (antepulsión).
                 **Alertas:** Cifo-lordosis acentuada, FHP, hombros protruídos, inclinación pélvica marcada.
            """,
            "metricas_clave": [
                {"nombre": "Postura de Cabeza (CVA)", "clave_resultado": "angulo_CVA", "norma_min": 47.0, "norma_max": 50.0, "interpretacion": "Bajo = FHP."},
                {"nombre": "Alineación Vertical (BL)", "clave_resultado": "inclinacion_corporal_v", "norma_min": -2.0, "norma_max": 2.0, "interpretacion": "Inclinación Tobillo-Hombro."},
                {"nombre": "Inclinación Pélvica (Proxy)", "clave_resultado": "inclinacion_pelvica_v", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": "Inclinación Hombro-Cadera."}
            ],
            "metricas_secundarias": [
                {"nombre": "Hiperextensión Rodilla", "clave_resultado": "extension_rodilla", "norma_min": 178.0, "norma_max": 182.0, "interpretacion": ">182° Genu Recurvatum."}
            ]
        },
         3: { # VISTA POSTERIOR ESTÁTICA
            "significado_biomecanico": "Evaluar la simetría de la espalda, la posición de las escápulas y la alineación del retropié (calcáneo).",
            "descripcion_completa": """
                **Nombre Técnico:** Evaluación Postural Estática – Plano Frontal (Posterior).
                **Key Landmarks:** Hombros (11, 12), Tobillos (27, 28), Talones (29, 30).
                **Análisis:**
                - **Alineación Calcáneo:** Ángulo (Tobillo-Talón) (ej. 28-30) vs. Vertical. **Norma:** 0° ± 5°. Valgo > 5° (pronación), Varo < -5° (supinación).
                - **Posición Escapular:** Observación cualitativa: distancia borde medial a columna, simetría, "aleteo" (winged scapula).
                **Alertas:** Escápulas aladas, asimetría escapular, valgo/varo calcáneo excesivo.
            """,
            "metricas_clave": [
                {"nombre": "Alineación Calcáneo (Der)", "clave_resultado": "angulo_talon_r_v", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": ">5° Valgo / <-5° Varo (Der)."},
                {"nombre": "Alineación Calcáneo (Izq)", "clave_resultado": "angulo_talon_l_v", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": ">5° Valgo / <-5° Varo (Izq)."}
            ],
            "metricas_secundarias": [
                {"nombre": "Nivel de Hombros", "clave_resultado": "angulo_hombros_h", "norma_min": -2.0, "norma_max": 2.0, "interpretacion": "Asimetría hombros."},
                {"nombre": "Oblicuidad Pélvica", "clave_resultado": "angulo_pelvis_h", "norma_min": -2.0, "norma_max": 2.0, "interpretacion": "Asimetría pélvica."}
            ]
        },
        4: { # OHS FRONTAL
            "significado_biomecanico": "Evaluar el control neuromuscular en el plano frontal durante un patrón de triple flexión bajo carga, identificando valgo dinámico, shift pélvico y estabilidad del pie.",
            "descripcion_completa": """
                **Nombre Técnico:** Overhead Squat Assessment – Plano Frontal (Fase Excéntrica Máxima).
                **Key Landmarks:** Caderas (23, 24), Rodillas (25, 26), Tobillos (27, 28), Puntas Pie (31, 32).
                **Análisis:**
                - **Valgo Dinámico (DKV):** Ángulo (Cadera-Rodilla-Tobillo) (ej. 24-26-28). **Norma:** <170° indica valgo (>10°). Rodilla no debe pasar medialmente línea Cadera-2º dedo. Riesgo lesión LCA.
                - **Shift Pélvico:** Desviación horizontal centro Caderas (23-24) vs. centro Tobillos (27-28). **Norma:** < 3 cm (aprox < 30-40px). Indica carga asimétrica.
                - **Pronación Dinámica:** Observar colapso arco medial / eversión tobillo (28 vs 32).
                **Alertas:** Colapso valgo (uni/bilateral), shift lateral, pronación excesiva.
            """,
            "metricas_clave": [
                {"nombre": "Valgo Dinámico (Der)", "clave_resultado": "angulo_FPPA_dinamico_der", "norma_min": 170.0, "norma_max": 180.0, "interpretacion": "<170° Valgo Derecho."},
                {"nombre": "Valgo Dinámico (Izq)", "clave_resultado": "angulo_FPPA_dinamico_izq", "norma_min": 170.0, "norma_max": 180.0, "interpretacion": "<170° Valgo Izquierdo."}
            ],
            "metricas_secundarias": [
                 {"nombre": "Shift Pélvico (px)", "clave_resultado": "desplazamiento_pelvico_px", "norma_min": -40.0, "norma_max": 40.0, "interpretacion": "Desplazamiento lateral (px)."} # Aumentado umbral px
            ]
        },
        5: { # OHS SAGITAL
            "significado_biomecanico": "Evaluar profundidad, movilidad sagital (dorsiflexión, flexión cadera/rodilla), estabilidad del core y movilidad escapulo-torácica.",
             "descripcion_completa": """
                **Nombre Técnico:** Overhead Squat Assessment – Plano Sagital (Fase Excéntrica Máxima).
                **Key Landmarks:** Oreja (8), Hombro (12), Cadera (24), Rodilla (26), Tobillo (28), Punta Pie (32) (lado visible).
                **Análisis:**
                - **Profundidad:** Ángulo flexión Cadera (12-24-26) y Rodilla (24-26-28). **Norma:** Cadera (24) bajo Rodilla (26).
                - **Inclinación Tronco vs. Tibia (Paralelismo):** Ángulo (12-24) vs. Vertical comparado con (26-28) vs. Vertical. **Norma:** Paralelos (dif < 10°). Tronco muy inclinado sugiere core/extensores débiles o tobillo limitado.
                - **Dorsiflexión Tobillo:** Ángulo (Tibia-Pie) (26-28-32). **Norma:** > 35-40° requeridos.
                - **Posición Brazos:** Húmero (12) alineado o detrás de Oreja (8). Caída indica restricción.
                - **Curvatura Lumbar:** Evitar flexión excesiva ("butt wink").
                 **Alertas:** Profundidad limitada, tronco excesivamente inclinado, talones elevados, brazos caídos, "butt wink".
            """,
            "metricas_clave": [
                {"nombre": "Paralelismo Tronco-Tibia", "clave_resultado": "paralelismo_tronco_tibia", "norma_min": -10.0, "norma_max": 10.0, "interpretacion": ">10° Dom.Cadera / <-10° Dom.Rodilla."},
                {"nombre": "Dorsiflexión (Proxy Tibia)", "clave_resultado": "inclinacion_tibia_v", "norma_min": 35.0, "norma_max": 50.0, "interpretacion": "Inclinación Tibia vs Vert. <35° limitado."},
                {"nombre": "Profundidad (Flex. Cadera)", "clave_resultado": "flexion_cadera", "norma_min": 60.0, "norma_max": 80.0, "interpretacion": "Ángulo interno. <80° profundo."}
            ],
            "metricas_secundarias": [
                {"nombre": "Inclinación Tronco", "clave_resultado": "inclinacion_tronco_v", "norma_min": 35.0, "norma_max": 50.0, "interpretacion": "Inclinación vs vertical."},
                {"nombre": "Profundidad (Flex. Rodilla)", "clave_resultado": "flexion_rodilla", "norma_min": 60.0, "norma_max": 80.0, "interpretacion": "Ángulo interno rodilla."}
            ]
        },
        6: { # OHS POSTERIOR
            "significado_biomecanico": "Confirmar shift pélvico, pronación dinámica del retropié (valgo calcáneo) y detectar elevación de talones.",
            "descripcion_completa": """
                **Nombre Técnico:** Overhead Squat Assessment – Plano Frontal (Posterior, Fase Excéntrica Máxima).
                **Key Landmarks:** Caderas (23, 24), Tobillos (27, 28), Talones (29, 30).
                **Análisis:**
                - **Valgo Dinámico Calcáneo:** Ángulo (Tobillo-Talón) (ej. 28-30) vs. Vertical. **Norma:** 0° a 7°. > 7° indica hiperpronación bajo carga.
                - **Elevación Talones:** Distancia vertical (29, 30) al suelo. **Norma:** Cero. Indica limitación dorsiflexión.
                - **Shift Pélvico:** Confirmación visual/cuantitativa desplazamiento centro Caderas (23-24).
                - **Simetría Brazos:** Observación cualitativa.
                **Alertas:** Shift confirmado, valgo calcáneo excesivo, talones elevados, asimetría brazos.
            """,
            "metricas_clave": [
                {"nombre": "Valgo Calcáneo (Der)", "clave_resultado": "angulo_talon_r_v", "norma_min": -7.0, "norma_max": 7.0, "interpretacion": "Valgo/Pronación dinámica derecha."},
                {"nombre": "Valgo Calcáneo (Izq)", "clave_resultado": "angulo_talon_l_v", "norma_min": -7.0, "norma_max": 7.0, "interpretacion": "Valgo/Pronación dinámica izquierda."}
            ],
            "metricas_secundarias": [
                {"nombre": "Shift Pélvico (px)", "clave_resultado": "desplazamiento_pelvico_px", "norma_min": -40.0, "norma_max": 40.0, "interpretacion": "Confirmación desplazamiento."}
            ]
        },
        7: { # SLS IZQUIERDA
            "significado_biomecanico": "Evaluar la estabilidad y control neuromuscular del miembro inferior izquierdo (cadera, rodilla, tobillo) en condiciones unipodales.",
            "descripcion_completa": """
                **Nombre Técnico:** Single Leg Squat Assessment – Plano Frontal (Pierna Izquierda Apoyada).
                **Key Landmarks:** Hombros (11, 12), Caderas (23, 24), Rodilla Izq (25), Tobillo Izq (27), Pie Izq (31).
                **Análisis:**
                - **Valgo Dinámico Unipodal:** Ángulo (23-25-27). **Norma:** Mínimo colapso (<10-15° valgo / >165-170°).
                - **Caída Pélvica (Pelvic Drop):** Ángulo (23-24) vs. Horizontal. **Norma:** Caída < 5° del lado derecho (libre). Indica debilidad Glúteo Medio izquierdo.
                - **Compensación Tronco:** Inclinación lateral (11-12) hacia la izquierda. **Norma:** Mínima.
                - **Estabilidad Tobillo/Pie:** Observar oscilaciones / colapso arco (27-31).
                **Alertas:** Valgo rodilla, caída pélvica, inclinación tronco, inestabilidad tobillo.
            """,
            "metricas_clave": [
                {"nombre": "Valgo Dinámico (Izq)", "clave_resultado": "angulo_FPPA_sls_izq", "norma_min": 165.0, "norma_max": 180.0, "interpretacion": "Valgo izquierdo >15° (<165°)."}, # Norma más estricta para SLS
                {"nombre": "Caída Pélvica (CPD)", "clave_resultado": "angulo_CPD", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": "Caída contralateral <5°."}
            ],
            "metricas_secundarias": [
                {"nombre": "Inclinación Tronco", "clave_resultado": "inclinacion_tronco_v", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": "Compensación lateral."}
            ]
        },
        8: { # SLS DERECHA
            "significado_biomecanico": "Evaluar la estabilidad y control neuromuscular del miembro inferior derecho y comparar con la izquierda para cuantificar la asimetría funcional.",
            "descripcion_completa": """
                **Nombre Técnico:** Single Leg Squat Assessment – Plano Frontal (Pierna Derecha Apoyada).
                **Key Landmarks:** Hombros (11, 12), Caderas (23, 24), Rodilla Der (26), Tobillo Der (28), Pie Der (32).
                **Análisis:** Comparar los siguientes valores con la Pose 7.
                - **Valgo Dinámico Unipodal:** Ángulo (24-26-28). **Norma:** <10-15° valgo / >165-170°.
                - **Caída Pélvica (Pelvic Drop):** Ángulo (23-24) vs. Horizontal. **Norma:** Caída < 5° del lado izquierdo (libre).
                - **Compensación Tronco:** Inclinación lateral (11-12) hacia la derecha. **Norma:** Mínima.
                - **Índice Asimetría:** Diferencia (%) en valgo, caída pélvica o profundidad vs. Pose 7. >10-15% es significativo.
                **Alertas:** Valgo, caída pélvica, inclinación tronco (mayores que en lado izq). Asimetrías significativas.
            """,
            "metricas_clave": [
                {"nombre": "Valgo Dinámico (Der)", "clave_resultado": "angulo_FPPA_sls_der", "norma_min": 165.0, "norma_max": 180.0, "interpretacion": "Valgo derecho >15° (<165°)."}, # Norma más estricta para SLS
                {"nombre": "Caída Pélvica (CPD)", "clave_resultado": "angulo_CPD", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": "Caída contralateral <5°."}
            ],
            "metricas_secundarias": [
                {"nombre": "Inclinación Tronco", "clave_resultado": "inclinacion_tronco_v", "norma_min": -5.0, "norma_max": 5.0, "interpretacion": "Compensación lateral."}
            ]
        }
    }
    default_explanation = {"significado_biomecanico": "N/A", "descripcion_completa": "N/A", "metricas_clave": [], "metricas_secundarias": []}
    explanation = explanations.get(pose_index, default_explanation)
    for key, value in default_explanation.items():
        if key not in explanation: explanation[key] = value
    return explanation

# -----------------------------------------------------------------------------
# 5. FUNCIONES DE RENDERIZADO DE VISTAS (Interfaz)
# -----------------------------------------------------------------------------

def render_login_view(users_df):
    """Muestra la vista de Login de forma compacta."""
    # (Sin cambios)
    _, col_main, _ = st.columns([1, 1, 1])
    with col_main:
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=120)
        except Exception as e: st.error(f"Error logo: {e}")
        st.subheader("Análisis Biomecánico CAB")
        with st.form("login_form"):
            username = st.text_input("Usuario", label_visibility="collapsed", placeholder="Usuario")
            password = st.text_input("Contraseña", type="password", label_visibility="collapsed", placeholder="Contraseña")
            submitted = st.form_submit_button("Ingresar")
            if submitted:
                if check_login(username, password, users_df):
                    st.session_state.logged_in = True; st.session_state.username = username
                    st.session_state.vista_actual = "menu"; st.rerun()
                else: st.error("Usuario o contraseña incorrectos.")

def render_menu_view():
    """Muestra el Menú Principal con el título como botón."""
    # (Sin cambios)

    # --- Barra Superior ---
    col_main, col_logout = st.columns([0.85, 0.15])

    with col_main:
        col_logo_inner, col_title_inner = st.columns([0.1, 0.9])
        with col_logo_inner:
            try:
                logo = Image.open(LOGO_PATH)
                st.image(logo, width=100)
            except Exception as e:
                st.error("Error logo")

        with col_title_inner:
            st.subheader("Análisis Biomecánico Postural")
            st.markdown("""
            <p style="font-size: 1.05rem; margin-top: -8px; color: #FAFAFA;">
            Plataforma de análisis biomecánico que utiliza Computer Vision
            para cuantificar la alineación postural y la cinemática del movimiento.
            Seleccione un protocolo para evaluar al atleta.
            </p>
            """, unsafe_allow_html=True)

    with col_logout:
        if st.button("Cerrar Sesión", key="logout_menu", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False; st.session_state.username = None
            st.session_state.vista_actual = "login";
            if 'pose_seleccionada_info' in st.session_state: del st.session_state['pose_seleccionada_info']
            keys_to_clear = [k for k in st.session_state if k.startswith('uploader_') or k.startswith('processed_') or k.startswith('analysis_') or k.startswith('last_uploaded_')]
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()

    st.markdown("""<hr style="margin-top: 0.5rem; margin-bottom: 0.5rem;" /> """, unsafe_allow_html=True)

    # --- Grid 2x4 ---
    pose_keys = sorted(POSE_FILES_INFO.keys(), key=lambda x: POSE_FILES_INFO[x]['index'])
    idx = 0
    num_cols = 4
    num_rows = (len(pose_keys) + num_cols - 1) // num_cols

    for row in range(num_rows):
        cols = st.columns(num_cols, gap="small")
        for col_idx in range(num_cols):
            if idx < len(pose_keys):
                with cols[col_idx]:
                    file_key = pose_keys[idx]
                    pose_info = POSE_FILES_INFO[file_key]
                    pose_title = pose_info["title"]
                    pose_index = pose_info["index"]
                    img_filename = f"{file_key}.png"
                    img_path = os.path.join(POSE_IMG_DIR, img_filename)

                    try:
                        st.markdown(f'<div class="pose-container">', unsafe_allow_html=True)

                        button_key = f"pose_button_{pose_index}"
                        if st.button(pose_title, key=button_key, use_container_width=True):
                                st.session_state.vista_actual = f"pose_{pose_index}"
                                st.session_state.pose_seleccionada_info = {
                                    "file_key": file_key, "title": pose_title, "index": pose_index
                                }
                                st.rerun()

                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                    except FileNotFoundError: st.error(f"Falta:\n{img_filename}")
                    except UnidentifiedImageError: st.error(f"Formato:\n{img_filename}")
                    except Exception as e: st.error(f"Error:\n{img_filename}")
            idx += 1


def render_pose_detail_view():
    """
    CORREGIDO: Lógica de procesamiento reinsertada y aislada para garantizar el flujo.
    """
    if "pose_seleccionada_info" not in st.session_state or st.session_state.pose_seleccionada_info is None:
        st.warning("..."); st.session_state.vista_actual = "menu"; st.rerun(); return

    pose_info = st.session_state.pose_seleccionada_info
    pose_index = pose_info["index"]
    pose_title = pose_info["title"]
    file_key = pose_info["file_key"]

    # Keys necesarias para el estado
    processed_bytes_key = f'processed_bytes_{pose_index}'
    analysis_text_key = f'analysis_text_{pose_index}'
    analysis_angles_key = f'analysis_angles_{pose_index}'
    analysis_msg_key = f'analysis_msg_{pose_index}'
    uploader_key = f"uploader_{pose_index}" # Clave del uploader

    # --- LÓGICA CRÍTICA: Procesamiento del archivo ---
    # Recuperar el valor del uploader después de la ejecución del ciclo anterior
    # Importante: Streamlit lee el valor del uploader antes de que se renderice completamente en el ciclo.

    # Intentar obtener el archivo subido del estado (si ya existe)
    uploaded_file = st.session_state.get(uploader_key)

    if uploaded_file is not None:
        # Verificar si el archivo es nuevo o no ha sido procesado
        if f'last_uploaded_id_{pose_index}' not in st.session_state or st.session_state[f'last_uploaded_id_{pose_index}'] != uploaded_file.file_id:
            with st.spinner("Analizando postura..."):
                image_data = uploaded_file.getvalue()
                skeleton_bytes, analysis_string, analysis_dict = process_uploaded_image(image_data, pose_index)

                st.session_state[processed_bytes_key] = skeleton_bytes
                st.session_state[analysis_text_key] = analysis_string
                st.session_state[analysis_angles_key] = analysis_dict
                st.session_state[f'last_uploaded_id_{pose_index}'] = uploaded_file.file_id

                # Guardar datos para LSI
                if pose_index == 7: st.session_state['profundidad_sls_izq'] = analysis_dict.get('flexion_rodilla_sls_izq')
                elif pose_index == 8: st.session_state['profundidad_sls_der'] = analysis_dict.get('flexion_rodilla_sls_der')

                st.session_state[analysis_msg_key] = "Análisis completado." if skeleton_bytes else analysis_string

                st.rerun() # Disparar el re-renderizado

    # --- FIN DE LA LÓGICA CRÍTICA ---


    # --- Barra Superior (Renderizado) ---
    col_back, col_title_spacer, col_logout_detail = st.columns([0.1, 0.65, 0.25])

    with col_back:
        if st.button("Volver", use_container_width=True):
            st.session_state.vista_actual = "menu"; st.rerun()
    with col_title_spacer:
        pass
    with col_logout_detail:
        st.markdown('<div style="display: flex; justify-content: flex-end;">', unsafe_allow_html=True)
        if st.button("Cerrar Sesión", key="logout_detail", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False; st.session_state.username = None
            st.session_state.vista_actual = "login"; st.session_state.pose_seleccionada_info = None
            keys_to_clear = [k for k in st.session_state if k.startswith('uploader_') or k.startswith('processed_') or k.startswith('analysis_') or k.startswith('last_uploaded_')]
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""<hr style="margin-top: 0.5rem; margin-bottom: 1rem;" /> """, unsafe_allow_html=True)

    # --- FILA SUPERIOR: Contexto (Renderizado) ---


    col_info_img, col_info_text = st.columns([0.4, 0.6], gap="large")

    with col_info_img:
        # CUADRANTE 1: Imagen de Referencia (Sin título)
        img_filename = f"{file_key}.png"
        img_path = os.path.join(POSE_IMG_DIR, img_filename)
        try:
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.error(f"No se encontró la imagen de ejemplo: {img_filename}")

    with col_info_text:
        # CUADRANTE 2: Título y Protocolo
        st.subheader(pose_title)
        explanation_dict = get_explanation_for_pose(pose_index)
        significado_text = explanation_dict.get("significado_biomecanico", "N/A")
        protocolo_text = explanation_dict.get("descripcion_completa", "N/A")

        st.markdown("##### Significado Biomecánico")
        st.markdown(significado_text, unsafe_allow_html=True)

        with st.expander("Ver Protocolo Detallado y Métricas Clave"):
            st.markdown(protocolo_text, unsafe_allow_html=True)

    st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 1.5rem;" /> """, unsafe_allow_html=True)

    # --- FILA INFERIOR: Acción y Resultados (Renderizado) ---
    col_result_display, col_metrics = st.columns([0.4, 0.6], gap="large")

    # Obtener el estado de procesamiento
    image_processed = processed_bytes_key in st.session_state and st.session_state[processed_bytes_key] is not None


    # --- CUADRANTE 4: Resultados Numéricos (Renderizado) ---
    with col_metrics:
        st.subheader("Resultados Cuantitativos")

        explanation_dict = get_explanation_for_pose(pose_index)
        resultados_numericos = st.session_state.get(analysis_angles_key, {})

        if not image_processed:
            st.info("Los resultados numéricos aparecerán aquí después de procesar la imagen.")

        # Renderizar Métricas Clave
        for metrica in explanation_dict.get("metricas_clave", []):
            nombre = metrica["nombre"]
            clave = metrica["clave_resultado"]
            valor = resultados_numericos.get(clave)

            if not image_processed:
                st.metric(label=nombre, value="--", delta="Esperando imagen", delta_color="off")
            elif valor is not None:
                norma_min = metrica["norma_min"]
                norma_max = metrica["norma_max"]
                delta_str = f"Norma: {norma_min:.1f}° - {norma_max:.1f}°"
                delta_color = "normal"
                if valor < norma_min or valor > norma_max: delta_color = "inverse"
                st.metric(label=nombre, value=f"{valor:.1f}°", delta=delta_str, delta_color=delta_color)
                with st.expander(f"Análisis de {nombre}"):
                    st.markdown(f"**KPI:** {metrica['interpretacion']}")
                    if "ciencia" in metrica: st.caption(f"**Contexto:** {metrica['ciencia']}")
            else:
                st.metric(label=nombre, value="N/A", delta="No calculado", delta_color="off")

        # Lógica de LSI
        if (pose_index == 7 or pose_index == 8):
            st.markdown("##### Índice de Simetría (LSI)")
            if image_processed:
                prof_izq = st.session_state.get('profundidad_sls_izq')
                prof_der = st.session_state.get('profundidad_sls_der')

                if prof_izq is not None and prof_der is not None:
                    if prof_der == 0: prof_der = 1e-6
                    if prof_izq == 0: prof_izq = 1e-6
                    lsi_izq_der = (prof_izq / prof_der) * 100
                    lsi_der_izq = (prof_der / prof_izq) * 100
                    st.info("LSI basado en la profundidad (ángulo de flexión de rodilla). Objetivo > 90%.")
                    col_lsi1, col_lsi2 = st.columns(2)
                    col_lsi1.metric(label="LSI (Izq vs Der)", value=f"{lsi_izq_der:.1f}%", delta=f"{prof_izq:.1f}° / {prof_der:.1f}°")
                    col_lsi2.metric(label="LSI (Der vs Izq)", value=f"{lsi_der_izq:.1f}%", delta=f"{prof_der:.1f}° / {prof_izq:.1f}°")
                else:
                    st.info("Analiza ambas piernas (Pose 7 y 8) para calcular el LSI.")
            else:
                 st.info("El LSI se calculará después de analizar ambas piernas.")


    # --- CUADRANTE 3: Imagen Procesada y Carga (Renderizado) ---
    with col_result_display:

        if image_processed:
            st.image(st.session_state[processed_bytes_key], use_container_width=True)

            # Bloque de Descarga
            st.markdown("""<div style="margin-top: 1rem;"></div>""", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button( "📥 Descargar Imagen", st.session_state[processed_bytes_key], f"esqueleto_{file_key}.png", "image/png", use_container_width=True, type="primary")
            with col_btn2:
                analysis_content = st.session_state.get(analysis_text_key, "").encode('utf-8')
                st.download_button("📄 Descargar Análisis", analysis_content, f"analisis_{file_key}.txt", "text/plain", use_container_width=True, type="secondary", disabled=(not analysis_content))

            # Bloque de Carga (Abajo de todo)
            st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;" /> """, unsafe_allow_html=True)
            st.markdown("##### Cargar nueva imagen")

        else:
            st.info("La imagen procesada aparecerá aquí.")

        # Uploader (Renderizado Abajo de todo)
        # Note: Ya lo definimos al principio, aquí solo lo renderizamos en su lugar final.
        st.file_uploader("Cargar imagen para analizar:", type=["jpg", "jpeg", "png"],
                         key=uploader_key, label_visibility="collapsed")

# -----------------------------------------------------------------------------
# 6. LÓGICA PRINCIPAL DE LA APLICACIÓN (MAIN)
# -----------------------------------------------------------------------------
def main():
    users_df = get_users(USERS_CSV_PATH)
    if users_df is None: st.stop()

    # Inicializar estado de sesión
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'vista_actual' not in st.session_state: st.session_state.vista_actual = "login"
    if 'username' not in st.session_state: st.session_state.username = None
    if 'pose_seleccionada_info' not in st.session_state: st.session_state.pose_seleccionada_info = None

    # Router
    current_view = st.session_state.get('vista_actual', 'login')

    if not st.session_state.get('logged_in', False) or current_view == "login":
        # Forzar logout state
        st.session_state.logged_in = False; st.session_state.username = None
        st.session_state.vista_actual = "login"; st.session_state.pose_seleccionada_info = None
        keys_to_clear = [k for k in st.session_state if k.startswith('uploader_') or k.startswith('processed_') or k.startswith('analysis_') or k.startswith('last_uploaded_')]
        for key in keys_to_clear: del st.session_state[key]
        render_login_view(users_df)
    elif current_view == "menu":
        render_menu_view()
    elif current_view.startswith("pose_"):
        render_pose_detail_view()
    else: # Estado inválido
        st.warning("Estado desconocido."); st.session_state.vista_actual = "login"; st.rerun()

# -----------------------------------------------------------------------------
# 7. EJECUCIÓN DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    # Cerrar detector de pose al finalizar (importante si no es static_image_mode=True)
    # Si es True, no es estrictamente necesario, pero buena práctica.
    # Considerar mover pose_detector a st.session_state o usar @st.cache_resource si se usa fuera de process_uploaded_image
    # pose_detector.close() # Comentado por ahora, ya que se inicializa globalmente
