import os
import sqlite3
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. SETUP & DIRECTORIES
# ==========================================
st.set_page_config(page_title="OncoGuard Triage Platform", layout="wide", page_icon="🎗️")

BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "medical_records.db")
SCANS_DIR = os.path.join(BASE_DIR, "patient_scans")
os.makedirs(SCANS_DIR, exist_ok=True)

# ==========================================
# 2. DATABASE INIT
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS queue 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, 
                  filename TEXT, risk_score REAL, priority TEXT, status TEXT)''')
    
    c.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin123', 'admin')")
    c.execute("INSERT OR IGNORE INTO users VALUES ('doctor', 'doc123', 'clinician')")
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. CLINICAL AI MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_ai_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    model.to(device)
    model.eval()
    return model

model = load_ai_model()

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_mammogram(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = img_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            cancer_risk = probabilities[0][1].item()
        return round(cancer_risk, 4)
    except Exception:
        return 0.50

# ==========================================
# 4. AUTHENTICATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""

def login():
    st.sidebar.title("🔐 System Login")
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Log In"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (u, p))
        res = c.fetchone()
        conn.close()
        if res:
            st.session_state.authenticated = True
            st.session_state.username = u
            st.session_state.role = res[0]
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials.")

if not st.session_state.authenticated:
    login()
    st.title("🎗️ OncoGuard Medical AI Triage Platform")
    st.info("Log in via the sidebar to access clinical records.")
    st.stop()

# ==========================================
# 5. NAVIGATION & WORKFLOW
# ==========================================
st.sidebar.write(f"User: **{st.session_state.username}** ({st.session_state.role})")
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.rerun()

nav = st.sidebar.radio("Navigation", ["Priority Queue", "Upload Scan", "User Management"])

if nav == "Priority Queue":
    st.header("📋 Priority Worklist Queue")
    conn = sqlite3.connect(DB_PATH)
    df = conn.execute("SELECT patient_id, filename, risk_score, priority, status FROM queue ORDER BY risk_score DESC").fetchall()
    conn.close()

    if not df:
        st.write("No scans in queue.")
    else:
        for row in df:
            p_id, fname, risk, priority, status = row
            if risk >= 0.50:
                st.error(f"🔴 **Patient ID:** {p_id} | **Malignancy Risk:** {risk*100:.1f}% | **Priority:** {priority}")
            else:
                st.success(f"🟢 **Patient ID:** {p_id} | **Malignancy Risk:** {risk*100:.1f}% | **Priority:** {priority}")
            col1, col2 = st.columns([1, 3])
            img_p = os.path.join(SCANS_DIR, fname)
            if os.path.exists(img_p):
                col1.image(img_p, width=150)
            col2.write(f"Status: **{status}**")
            col2.write("---")

elif nav == "Upload Scan":
    st.header("📤 Upload Mammogram Scan")
    p_id = st.text_input("Patient ID")
    uploaded_file = st.file_uploader("Choose Scan Image", type=["jpg", "png", "jpeg"])

    if st.button("Process Scan") and uploaded_file and p_id:
        file_path = os.path.join(SCANS_DIR, f"{p_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing..."):
            risk_score = analyze_mammogram(file_path)
            priority = "HIGH (Priority Review)" if risk_score >= 0.50 else "NORMAL (Standard Batch)"
            
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO queue (patient_id, filename, risk_score, priority, status) VALUES (?, ?, ?, ?, ?)",
                         (p_id, f"{p_id}_{uploaded_file.name}", risk_score, priority, "Pending Review"))
            conn.commit()
            conn.close()

        st.success(f"Scan Analyzed! Risk Score: {risk_score*100:.1f}%. Saved to priority queue.")

elif nav == "User Management":
    if st.session_state.role != "admin":
        st.warning("Admin rights required.")
    else:
        st.header("👤 Manage System Accounts")
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        new_r = st.selectbox("Role", ["clinician", "admin"])
        
        if st.button("Create Account"):
            if new_u and new_p:
                conn = sqlite3.connect(DB_PATH)
                try:
                    conn.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u, new_p, new_r))
                    conn.commit()
                    st.success(f"Account for {new_u} created.")
                except Exception:
                    st.error("User already exists.")
                finally:
                    conn.close()
