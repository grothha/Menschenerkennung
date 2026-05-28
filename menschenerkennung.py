import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Schuppen-Licht", page_icon="💡", layout="centered")

st.title("💡 KI-gestützte Lichtsteuerung")
st.markdown("Diese App analysiert Videostreams und entscheidet, ob das Schuppenlicht eingeschaltet wird.")

# --- KI MODELL LADEN ---
# Wir nutzen das kleinste YOLOv8-Modell (nano), da es extrem schnell ist und perfekt für Streamlit passt.
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- INTERFACE ---
st.sidebar.header("⚙️ Einstellungen")
confidence_threshold = st.sidebar.slider("KI-Sicherheit (Confidence Threshold)", 0.0, 1.0, 0.5, 0.05)

# Auswahl der Quelle
source_type = st.radio("Videoquelle auswählen:", ("Beispiel-Video hochladen", "Live IP-Kamera-Stream (RTSP/HTTP)"))

video_path = None

if source_type == "Beispiel-Video hochladen":
    uploaded_file = st.file_uploader("Wähle ein kurzes Garten-Video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Streamlit muss die Datei temporär speichern, damit OpenCV sie lesen kann
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

else:
    camera_url = st.text_input("IP-Kamera URL (z.B. http://192.168.1.50/video.mjpg oder rtsp://...)", "")
    if camera_url:
        video_path = camera_url

# --- VERARBEITUNG & LOGIK ---
if video_path:
    st.subheader("📺 Video-Analyse läuft...")
    
    # Platzhalter für das Live-Bild und den Status
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # OpenCV Video-Capture starten
    cap = cv2.VideoCapture(video_path)
    
    # Klassen-IDs in YOLOv8 für COCO-Datensatz: 
    # 0 = Person, 3 = Dog, 15 = Cat, 16 = Bird (Waschbären fallen oft unter "bird/cat/dog" oder "animal", falls nicht speziell nachtrainiert, YOLOv8 erkennt Menschen aber zu 100%)
    PERSON_CLASS_ID = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8 Objekterkennung auf dem aktuellen Frame ausführen
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Logik-Variable zurücksetzen
        mensch_erkannt = False
        waschbaer_oder_tier_erkannt = False
        
        # Ergebnisse auswerten
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                
                # Prüfen, ob ein Mensch im Bild ist
                if class_id == PERSON_CLASS_ID:
                    mensch_erkannt = True
                
                # Optionale Erkennung von anderen Tieren (für die Anzeige)
                if label in ["cat", "dog", "bird", "bear"]: # "bear" oder "cat" wird oft für Waschbären getriggert
                    waschbaer_oder_tier_erkannt = True
                    
            # Gezeichnetes Bild von YOLO holen
            frame = result.plot()
        
        # --- SMART LOGIK ---
        if mensch_erkannt:
            status_placeholder.success("🟢 MENSCH ERKANNT! -> 💡 LICHT EINSCHALTEN")
            # HIER KÖNNTE SPÄTER DER BEFEHL AN DEINE SMARTE LAMPE STEHEN
            # z.B. requests.get("http://<shelly-ip>/relay/0?turn=on")
        elif waschbaer_oder_tier_erkannt:
            status_placeholder.warning("🦝 Tier erkannt! -> ❌ Licht bleibt AUS (Fehlalarm verhindert)")
        else:
            status_placeholder.info("⚪ Keine Bewegung/Mensch -> ❌ Licht bleibt AUS")
            
        # OpenCV nutzt BGR, Streamlit braucht RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Frame in der App anzeigen
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Kleiner Delay, damit es flüssig läuft
        time.sleep(0.03)
        
    cap.release()
else:
    st.info("Bitte lade ein Video hoch oder gib eine Kamera-URL ein, um die KI-Steuerung zu testen.")
