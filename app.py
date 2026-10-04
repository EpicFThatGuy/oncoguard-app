import streamlit as st
import os
import sqlite3
import torch
import hashlib
from PIL import Image

# Import local backend modules
from model_architecture import DualBackboneOncoGuard, CalibratedInferenceEngine
from dataset_streamer import ShardedMammographyStreamer

# ==========================================
# 1. PAGE & DATABASE CONFIGURATION
# ==========================================
st.set_page_config(page_title="OncoGuard AI Platform", layout="wide", page_icon="🎗️")

BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "system_records.db")
SCANS_DIR = os.path.join(BASE_DIR, "patient_scans")
os.makedirs(SCANS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans 
                 (scan_id TEXT PRIMARY KEY, patient_id TEXT, uploader TEXT, 
                  risk_score REAL, priority TEXT, image_path TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Seed default admin
    pwd_hash = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users VALUES ('admin', ?, 'Admin')", (pwd_hash,))
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBackboneOncoGuard(pretrained=True)
    model.to(device)
    model.eval()
    return CalibratedInferenceEngine(model, platt_a=1.12, platt_b=-0.45), device

# ==========================================
# 2. AUTHENTICATION & SESSION STATE
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""

st.sidebar.title("🎗️ OncoGuard AI")

if not st.session_state.authenticated:
    st.subheader("Clinician Portal Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        pwd_hash = hashlib.sha256(p.encode()).hexdigest()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (u, pwd_hash))
        res = c.fetchone()
        conn.close()
        if res:
            st.session_state.authenticated = True
            st.session_state.username = u
            st.session_state.role = res[0]
            st.rerun()
        else:
            st.error("Invalid credentials.")
else:
    st.sidebar.write(f"Logged in: **{st.session_state.username}** ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    menu = ["Worklist Prioritization", "Submit Scan for Testing"]
    if st.session_state.role == "Admin":
        menu.append("User Management")
    
    choice = st.sidebar.selectbox("Navigation", menu)

    # ----------------------------------------------------
    # TRIAGE WORKLIST (INDEX 0 PRIORITY SORTING)
    # ----------------------------------------------------
    if choice == "Worklist Prioritization":
        st.header("📋 Specialist Worklist Queue (AI Priority Sorted)")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT patient_id, risk_score, priority, timestamp FROM scans ORDER BY risk_score DESC")
        scans = c.fetchall()
        conn.close()

        if not scans:
            st.info("No active patient scans in queue.")
        else:
            for patient_id, risk, priority, ts in scans:
                if risk >= 85.0:
                    st.error(f"🔴 **PATIENT: {patient_id}** | Malignancy Risk: {risk:.2f}% | Priority: {priority} | Time: {ts}")
                elif risk >= 35.0:
                    st.warning(f"🟡 **PATIENT: {patient_id}** | Malignancy Risk: {risk:.2f}% | Priority: {priority} | Time: {ts}")
                else:
                    st.success(f"🟢 **PATIENT: {patient_id}** | Malignancy Risk: {risk:.2f}% | Priority: {priority} | Time: {ts}")

    # ----------------------------------------------------
    # SUBMIT SCAN FOR INFERENCE
    # ----------------------------------------------------
    elif choice == "Submit Scan for Testing":
        st.header("📤 Upload Patient Mammogram")
        patient_id = st.text_input("Patient ID / Record Number")
        uploaded_file = st.file_uploader("Select Mammogram Image", type=["png", "jpg", "jpeg", "dcm"])

        if uploaded_file and patient_id:
            if st.button("Run AI Diagnostics & Route Case"):
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"Scan for Patient {patient_id}", width=300)
                
                engine, device = load_engine()
                streamer = ShardedMammographyStreamer([])
                
                # Preprocess via CLAHE
                bytes_data = uploaded_file.getvalue()
                pil_img = streamer.preprocess_clahe(bytes_data)
                
                if pil_img is None:
                    pil_img = image

                input_tensor = streamer.default_transform()(pil_img).unsqueeze(0).to(device)
                result = engine.predict(input_tensor)
                
                risk_pct = round(result["calibrated_probability"] * 100, 2)
                priority = result["priority_code"]

                # Save file locally
                file_path = os.path.join(SCANS_DIR, f"{patient_id}_{uploaded_file.name}")
                image.save(file_path)

                # Insert into DB
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                          (f"SCAN_{patient_id}", patient_id, st.session_state.username, risk_pct, priority, file_path))
                conn.commit()
                conn.close()

                st.success(f"Analysis Complete! Risk Score: {risk_pct}% | Status: {priority}")

    # ----------------------------------------------------
    # USER MANAGEMENT
    # ----------------------------------------------------
    elif choice == "User Management" and st.session_state.role == "Admin":
        st.header("👤 Manage System Users")
        nu = st.text_input("New Username")
        np_pass = st.text_input("New Password", type="password")
        nr = st.selectbox("Role", ["Doctor", "Patient", "Admin"])

        if st.button("Create Login Account"):
            if nu and np_pass:
                pwd_hash = hashlib.sha256(np_pass.encode()).hexdigest()
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO users VALUES (?, ?, ?)", (nu, pwd_hash, nr))
                    conn.commit()
                    st.success(f"Account for '{nu}' created.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")
                conn.close()
