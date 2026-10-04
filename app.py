import streamlit as st
import sqlite3
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import uuid
import pandas as pd
import hashlib

from model_architecture import DualBackboneOncoGuard, CalibratedInferenceEngine

# Page Layout Setup
st.set_page_config(page_title="OncoGuard AI - Medical Triage Workspace", layout="wide", page_icon="🩺")

# System Paths & Constants
BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "system_records.db")
SCANS_DIR = os.path.join(BASE_DIR, "patient_scans")
os.makedirs(SCANS_DIR, exist_ok=True)

# Hash password helper
def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database Initialization with Explicit Schema Safeguards
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, 
                    password TEXT NOT NULL, 
                    role TEXT NOT NULL)''')
    
    # Scans Table
    c.execute('''CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY, 
                    patient_id TEXT NOT NULL, 
                    uploaded_by TEXT NOT NULL, 
                    risk_score REAL NOT NULL, 
                    priority TEXT NOT NULL, 
                    pathology_type TEXT NOT NULL, 
                    birads_category TEXT NOT NULL, 
                    file_path TEXT NOT NULL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                    
    # Default Admin User
    admin_hash = hash_pass("admin123")
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES ('admin', ?, 'Admin')", (admin_hash,))
    
    conn.commit()
    conn.close()

init_db()

# Model Architecture Loader
@st.cache_resource
def load_oncology_engine():
    model = DualBackboneOncoGuard(pretrained=True)
    engine = CalibratedInferenceEngine(model=model)
    return engine

engine = load_oncology_engine()

# Transform Pipeline for Mammogram Ingestion
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Authentication Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# Authentication Gateway
if not st.session_state.authenticated:
    st.title("🩺 OncoGuard Enterprise Clinical Portal")
    st.subheader("Secure Oncology Diagnostic Terminal")
    
    with st.form("login_form"):
        username = st.text_input("Clinician Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Authenticate")
        
        if submit:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            row = c.fetchone()
            conn.close()
            
            if row and row[0] == hash_pass(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Authenticated successfully.")
                st.rerun()
            else:
                st.error("Invalid credentials. (Default: admin / admin123)")
    st.stop()

# Header Toolbar
col_title, col_user = st.columns([4, 1])
with col_title:
    st.title("🎗️ OncoGuard AI Workspace")
    st.caption("Deep Learning Mammography Triage & Pathology Prioritization System")
with col_user:
    st.write(f"Logged in: **{st.session_state.username}**")
    if st.button("Log Out"):
        st.session_state.authenticated = False
        st.rerun()

st.divider()

# Navigation Tabs
tab_scan, tab_workplace = st.tabs(["🔬 Run Scan Diagnostic", "📋 Workplace Prioritization Board"])

# -------------------------------------------------------------
# TAB 1: RUN SCAN DIAGNOSTIC
# -------------------------------------------------------------
with tab_scan:
    st.header("New Patient Diagnostic Ingestion")
    
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        patient_id = st.text_input("Patient Reference ID", placeholder="e.g. PT-90821")
        uploaded_file = st.file_uploader("Upload Digital Mammogram (JPG / PNG)", type=["jpg", "png", "jpeg"])
        run_btn = st.button("🚀 Execute Clinical Analysis", use_container_width=True)
    
    if run_btn:
        if not patient_id or not uploaded_file:
            st.warning("Please provide both a Patient Reference ID and a valid DICOM/Image scan.")
        else:
            # Generate Unique Storage Path & Unique Primary Key
            unique_scan_id = f"SCAN_{patient_id}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
            file_path = os.path.join(SCANS_DIR, f"{unique_scan_id}.png")
            
            # Save Image Locally
            image = Image.open(uploaded_file).convert("RGB")
            image.save(file_path)
            
            with col_preview:
                st.image(image, caption=f"Uploaded Scan: {patient_id}", use_column_width=True)
            
            # Run Inference Pipeline
            with st.spinner("Processing Dual-Backbone Feature Extractor & Platt Calibration..."):
                input_tensor = transform(image).unsqueeze(0)
                result = engine.predict(input_tensor)
                
            risk_pct = result["calibrated_probability"]
            priority = result["priority_code"]
            sla = result["target_sla"]
            birads = result["birads_category"]
            pathology = result["pathology_type"]
            clinical_note = result["clinical_note"]
            
            # Safe Database Insertion with Unique ID & Explicit Columns
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""INSERT INTO scans (scan_id, patient_id, uploaded_by, risk_score, priority, pathology_type, birads_category, file_path) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (unique_scan_id, patient_id, st.session_state.username, risk_pct, priority, pathology, birads, file_path))
            conn.commit()
            conn.close()
            
            # Display Clinical Results Card
            st.divider()
            st.subheader("📊 AI Diagnostic Output")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Malignancy Risk", f"{risk_pct}%")
            m2.metric("Triage Priority Tier", priority.split(":")[0])
            m3.metric("Target Clinical SLA", sla)
            
            if "DEFINITIVE MALIGNANCY" in priority:
                st.error(f"**Pathology Classification:** {pathology}")
                st.error(f"**BI-RADS Classification:** {birads}")
                st.error(f"**Clinical Action Note:** {clinical_note}")
            elif "SUSPICIOUS" in priority:
                st.warning(f"**Pathology Classification:** {pathology}")
                st.warning(f"**BI-RADS Classification:** {birads}")
                st.warning(f"**Clinical Action Note:** {clinical_note}")
            else:
                st.success(f"**Pathology Classification:** {pathology}")
                st.success(f"**BI-RADS Classification:** {birads}")
                st.success(f"**Clinical Action Note:** {clinical_note}")

# -------------------------------------------------------------
# TAB 2: WORKPLACE PRIORITIZATION BOARD
# -------------------------------------------------------------
with tab_workplace:
    st.header("📋 Priority Triage Queue")
    st.caption("Critical malignancy cases automatically float to the top for immediate specialist review.")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT scan_id AS 'Scan Reference', 
               patient_id AS 'Patient ID', 
               uploaded_by AS 'Clinician', 
               risk_score AS 'Malignancy Risk (%)', 
               priority AS 'Priority Tier', 
               pathology_type AS 'Pathology / Cancer Type', 
               birads_category AS 'BI-RADS Assessment', 
               timestamp AS 'Ingestion Timestamp' 
        FROM scans 
        ORDER BY risk_score DESC, timestamp DESC
    """, conn)
    conn.close()
    
    if df.empty:
        st.info("No active scans found in database. Upload a scan above to populate the queue.")
    else:
        # High Risk Filter Toggle
        show_urgent_only = st.checkbox("Show Priority 1 Urgent Cases Only")
        if show_urgent_only:
            df = df[df["Priority Tier"].str.contains("PRIORITY 1")]
            
        st.dataframe(
            df,
            column_config={
                "Malignancy Risk (%)": st.column_config.ProgressColumn(
                    "Malignancy Risk (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True,
            hide_index=True
        )
