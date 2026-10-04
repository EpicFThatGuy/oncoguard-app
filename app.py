import streamlit as st
import sqlite3
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import uuid
import pandas as pd
import numpy as np
import json
import hashlib

from model_architecture import (
    ConvNeXtClinicalExtractor,
    ClinicalImageProcessor,
    RadiomicPhysicsEngine,
    ClinicalReferenceBank
)

# =========================================================================
# 1. ENTERPRISE PACS THEME & LAYOUT CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="OncoGuard Enterprise PACS Workspace",
    layout="wide",
    page_icon="🎗️",
    initial_sidebar_state="expanded"
)

# Hospital Workstation Dark-Mode CSS Styling
st.markdown("""
<style>
    .main { background-color: #0b0e14; color: #e6edf3; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 12px; border-radius: 8px; }
    div[data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; }
    .badge-critical { background-color: #b91c1c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-urgent { background-color: #b45309; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-routine { background-color: #15803d; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Directory Structure Initialization
BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "clinical_pacs.db")
SCANS_DIR = os.path.join(BASE_DIR, "patient_repository")
os.makedirs(SCANS_DIR, exist_ok=True)

def hash_token(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =========================================================================
# 2. DATABASE LAYER WITH MIGRATION SAFEGUARDS
# =========================================================================
def init_pacs_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Clinician Authentication Registry
    c.execute('''CREATE TABLE IF NOT EXISTS clinicians (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    institution TEXT NOT NULL)''')
                    
    # 2. PACS Diagnostic Scans & Audit Warehouse
    c.execute('''CREATE TABLE IF NOT EXISTS pacs_scans (
                    scan_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    clinician_user TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    priority_tier TEXT NOT NULL,
                    birads_classification TEXT NOT NULL,
                    pathology_subtype TEXT NOT NULL,
                    matched_reference_id TEXT NOT NULL,
                    image_storage_path TEXT NOT NULL,
                    embedding_payload TEXT NOT NULL,
                    confirmed_ground_truth TEXT DEFAULT 'UNVERIFIED',
                    ingest_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Seed Default Master Oncologist Account
    master_pass = hash_token("admin123")
    c.execute("""INSERT OR IGNORE INTO clinicians (username, password_hash, role, institution) 
                 VALUES ('admin', ?, 'Chief Radiologist', 'Global Oncology Triage Hub')""", (master_pass,))
    
    conn.commit()
    conn.close()

init_pacs_db()

# =========================================================================
# 3. RESOURCE CACHING: DEEP MODELS & REFERENCE REPOSITORIES
# =========================================================================
@st.cache_resource
def load_system_engines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ConvNeXtClinicalExtractor().to(device).eval()
    ref_bank = ClinicalReferenceBank(extractor, device)
    
    # Hydrate reference bank from verified historical database entries
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT scan_id, embedding_payload, confirmed_ground_truth, pathology_subtype, birads_classification 
                 FROM pacs_scans WHERE confirmed_ground_truth IN ('MALIGNANT', 'BENIGN')""")
    rows = c.fetchall()
    conn.close()
    
    for sid, emb_json, gt, path, bi in rows:
        try:
            vec = json.loads(emb_json)
            ref_bank.register_case(sid, vec, gt, path, bi, "User-confirmed clinical ground truth")
        except Exception:
            continue
            
    return extractor, ref_bank, device

extractor, ref_bank, device = load_system_engines()

tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================================================================
# 4. CLINICIAN ACCESS CONTROL GATEWAY
# =========================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

if not st.session_state.authenticated:
    st.title("🛡️ OncoGuard Enterprise PACS Terminal")
    st.caption("Clinical-Grade Mammography Triage & Deep Reference Matcher")
    
    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        with st.form("auth_form"):
            st.subheader("Secure Practitioner Terminal Access")
            u = st.text_input("Practitioner ID / Username")
            p = st.text_input("Access Password", type="password")
            auth_btn = st.form_submit_button("Verify Identification", use_container_width=True)
            
            if auth_btn:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT password_hash, role FROM clinicians WHERE username = ?", (u,))
                row = c.fetchone()
                conn.close()
                
                if row and row[0] == hash_token(p):
                    st.session_state.authenticated = True
                    st.session_state.username = u
                    st.session_state.role = row[1]
                    st.success("Identity verified. Initializing workstation...")
                    st.rerun()
                else:
                    st.error("Authentication failed. (Default: admin / admin123)")
    st.stop()

# =========================================================================
# 5. WORKSTATION TOP BAR & NAVIGATION
# =========================================================================
header_c1, header_c2 = st.columns([4, 1])
with header_c1:
    st.title("🩺 OncoGuard PACS Workspace")
    st.caption(f"Authenticated Practitioner: **{st.session_state.username}** | Status: **{st.session_state.role}** | Reference Bank: **{len(ref_bank.reference_cases)} Clinical Archetypes Active**")
with header_c2:
    if st.button("Terminate Session", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.divider()

nav_tabs = st.tabs(["🔬 Patient Scan Ingestion", "📋 Urgent Worklist Queue", "🧠 Clinical Feedback & Active Memory"])

# =========================================================================
# TAB 1: SCAN INGESTION, PACS VIEWING & RADIOMIC EVALUATION
# =========================================================================
with nav_tabs[0]:
    st.header("Mammography Acquisition & Multi-Spectral Diagnostic View")
    
    col_up, col_info = st.columns([1, 2])
    
    with col_up:
        in_patient_id = st.text_input("Patient Identifier / Barcode Reference", placeholder="e.g. PT-99082")
        uploaded_file = st.file_uploader("Upload Digital Mammogram (DICOM / High-Res PNG / JPG)", type=["png", "jpg", "jpeg"])
        run_inference = st.button("🚀 Execute Comprehensive Analysis", use_container_width=True)

    if uploaded_file and in_patient_id:
        # Load raw file
        raw_pil = Image.open(uploaded_file)
        
        # 1. Execute Breast ROI Segmentation
        cropped_tissue = ClinicalImageProcessor.crop_to_breast_tissue(raw_pil)
        
        # 2. Generate PACS Multimodal Viewing Presets
        views = ClinicalImageProcessor.generate_view_presets(cropped_tissue)
        
        with col_info:
            view_mode = st.radio("PACS Window Preset:", ["Standard", "CLAHE Enhanced", "Inverted Film", "Calcification Focus"], horizontal=True)
            st.image(views[view_mode], caption=f"Patient {in_patient_id} - Mode: {view_mode}", use_container_width=True)
            
        if run_inference:
            with st.spinner("Extracting ConvNeXt Embeddings & Computing Reference Matrix..."):
                # Prepare Tensor
                input_tensor = tensor_transform(views["Standard"]).unsqueeze(0).to(device)
                
                # Extract 768-d Embedding Vector
                with torch.no_grad():
                    emb_tensor = extractor(input_tensor)
                    emb_vector = emb_tensor.squeeze(0).cpu().numpy().tolist()
                
                # Extract Physical Radiomics
                radiomics = RadiomicPhysicsEngine.extract_metrics(input_tensor)
                
                # Query Reference Knowledge Bank
                eval_res = ref_bank.query(emb_vector, radiomics)
                
                # Assign Safe Unique Primary Key for Database Insertion
                unique_scan_id = f"SCAN_{in_patient_id}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
                saved_path = os.path.join(SCANS_DIR, f"{unique_scan_id}.png")
                views["Standard"].save(saved_path)
                
                # Save to Persistent SQLite Database
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""INSERT INTO pacs_scans (scan_id, patient_id, clinician_user, risk_score, priority_tier, 
                                                     birads_classification, pathology_subtype, matched_reference_id, 
                                                     image_storage_path, embedding_payload) 
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (unique_scan_id, in_patient_id, st.session_state.username, eval_res["probability_pct"],
                           eval_res["priority"], eval_res["birads"], eval_res["pathology"], 
                           eval_res["top_match"]["ref_id"], saved_path, json.dumps(emb_vector)))
                conn.commit()
                conn.close()

            st.divider()
            st.subheader("📊 Definitive Triage Evaluation")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Calculated Malignancy Risk", f"{eval_res['probability_pct']}%")
            m2.metric("Triage Priority", eval_res["priority"].split(":")[0])
            m3.metric("BI-RADS Classification", eval_res["birads"])
            m4.metric("Diagnostic SLA", eval_res["target_sla"])
            
            # Clinical Findings Alert
            if eval_res["probability_pct"] >= 75.0:
                st.error(f"**Pathology Evaluation:** {eval_res['pathology']} | **Immediate Action Required:** {eval_res['top_match']['notes']}")
            elif eval_res["probability_pct"] >= 35.0:
                st.warning(f"**Pathology Evaluation:** {eval_res['pathology']} | **Audit Recommended:** {eval_res['top_match']['notes']}")
            else:
                st.success(f"**Pathology Evaluation:** {eval_res['pathology']} | **Negative Screening:** Routine follow-up.")

            # Side-by-Side Reference Comparison Panel
            st.subheader("🔗 Closest Matching Benchmark Case from Global Knowledge Bank")
            col_comp1, col_comp2 = st.columns([1, 1])
            with col_comp1:
                st.markdown("**Current Patient Scan**")
                st.image(views["CLAHE Enhanced"], use_container_width=True)
            with col_comp2:
                top_m = eval_res["top_match"]
                st.markdown(f"**Matched Clinical Reference ({top_m['ref_id']})**")
                st.info(f"**Dataset Archetype:** {top_m['ref_id']} \n\n"
                        f"**Confirmed Diagnosis:** {top_m['pathology']} \n\n"
                        f"**Cosine Correlation:** {top_m['similarity']:.4f} \n\n"
                        f"**Reference Pathology Manifest:** {top_m['notes']}")

# =========================================================================
# TAB 2: SPECIALIST TRIAGE WORKLIST (INDEX 0 PRIORITY SORTING)
# =========================================================================
with nav_tabs[1]:
    st.header("📋 Priority Radiology Worklist Queue")
    st.caption("Dynamic PACS worklist sorted by malignancy risk descending. The highest-risk patients sit permanently at Index 0.")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT scan_id AS 'Scan Reference',
               patient_id AS 'Patient ID',
               risk_score AS 'Malignancy Risk (%)',
               priority_tier AS 'Triage Priority',
               birads_classification AS 'BI-RADS Category',
               pathology_subtype AS 'Pathology Class',
               matched_reference_id AS 'Matched Benchmark Case',
               confirmed_ground_truth AS 'Ground Truth Verification',
               ingest_timestamp AS 'Time Logged'
        FROM pacs_scans
        ORDER BY risk_score DESC, ingest_timestamp DESC
    """, conn)
    conn.close()
    
    if df.empty:
        st.info("No active patient records logged in queue. Ingest a scan in Tab 1 to populate the worklist.")
    else:
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

# =========================================================================
# TAB 3: CONTINUOUS ACTIVE LEARNING (CLOSED-LOOP GROUND TRUTH ENGINE)
# =========================================================================
with nav_tabs[2]:
    st.header("🧠 Continuous Knowledge Expansion & Active Learning")
    st.caption("When a biopsy result or senior radiologist confirms an outcome, verify the case below. The system converts it into a permanent benchmark archetype.")
    
    conn = sqlite3.connect(DB_PATH)
    pending_df = pd.read_sql_query("SELECT scan_id, patient_id, risk_score, confirmed_ground_truth FROM pacs_scans ORDER BY ingest_timestamp DESC", conn)
    conn.close()
    
    if pending_df.empty:
        st.info("No scans available for active memory updates.")
    else:
        selected_scan_id = st.selectbox("Select Case to Verify Ground Truth:", pending_df["scan_id"].tolist())
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("✅ Confirm Ground Truth as BENIGN", use_container_width=True):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""UPDATE pacs_scans 
                             SET confirmed_ground_truth = 'BENIGN', 
                                 risk_score = 2.50, 
                                 priority_tier = '🟢 PRIORITY 3: ROUTINE BENIGN / NORMAL',
                                 birads_classification = 'BI-RADS 1-2 (Benign Confirmed)'
                             WHERE scan_id = ?""", (selected_scan_id,))
                
                # Fetch embedding to immediately inject into active memory
                c.execute("SELECT embedding_payload FROM pacs_scans WHERE scan_id = ?", (selected_scan_id,))
                row = c.fetchone()
                conn.commit()
                conn.close()
                
                if row:
                    vec = json.loads(row[0])
                    ref_bank.register_case(selected_scan_id, vec, "BENIGN", "Normal/Benign Confirmed", "BI-RADS 1", "Verified via Specialist Review")
                
                st.success(f"Scan {selected_scan_id} committed as verified BENIGN. Knowledge base updated!")
                st.rerun()
                
        with col_act2:
            if st.button("🚨 Confirm Ground Truth as MALIGNANT", use_container_width=True):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""UPDATE pacs_scans 
                             SET confirmed_ground_truth = 'MALIGNANT', 
                                 risk_score = 97.50, 
                                 priority_tier = '🔴 PRIORITY 1: DEFINITIVE MALIGNANCY (URGENT)',
                                 birads_classification = 'BI-RADS 5 (Biopsy Proven Malignant)'
                             WHERE scan_id = ?""", (selected_scan_id,))
                
                c.execute("SELECT embedding_payload FROM pacs_scans WHERE scan_id = ?", (selected_scan_id,))
                row = c.fetchone()
                conn.commit()
                conn.close()
                
                if row:
                    vec = json.loads(row[0])
                    ref_bank.register_case(selected_scan_id, vec, "MALIGNANT", "Biopsy Confirmed Carcinoma", "BI-RADS 5", "Verified via Histopathology")
                
                st.success(f"Scan {selected_scan_id} committed as verified MALIGNANT. Knowledge base updated!")
                st.rerun()
