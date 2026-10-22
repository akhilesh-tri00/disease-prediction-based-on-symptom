import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="MediPredict AI - Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional light green styling
st.markdown("""
<style>
    /* Professional Medical Light Green Theme */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    .card {
        background-color: #0f1f1a;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.25);
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #34d399;
    }
    
    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 1px solid #34d399 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
        box-shadow: 0 6px 20px rgba(52, 211, 153, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:not([kind="primary"]) {
        border-color: rgba(16, 185, 129, 0.3) !important;
        background-color: rgba(16, 185, 129, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:not([kind="primary"]):hover {
        border-color: #34d399 !important;
        color: #34d399 !important;
        background-color: rgba(16, 185, 129, 0.16) !important;
    }

    /* Active Tab indicator */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #34d399 !important;
        border-bottom-color: #10b981 !important;
    }

    /* Multiselect tags */
    span[data-baseweb="tag"] {
        background-color: rgba(16, 185, 129, 0.18) !important;
        border: 1px solid rgba(52, 211, 153, 0.4) !important;
        border-radius: 6px !important;
    }
    
    /* Metric Value */
    div[data-testid="stMetricValue"] {
        color: #34d399 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load('disease_model.pkl')
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)
    with open('disease_info.json', 'r') as f:
        info = json.load(f)
    return model, meta, info

try:
    model, metadata, disease_info = load_model_and_metadata()
except Exception as e:
    st.error(f"Error loading model resources: {e}. Please run `python train_and_evaluate.py` first.")
    st.stop()

# Header
st.markdown('<div class="main-header">MediPredict AI — Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Supervised Machine Learning Diagnostic Tool trained on 4,920 records across 41 conditions</div>', unsafe_allow_html=True)

# Tabs
tab_diag, tab_graphs, tab_info = st.tabs(["🩺 Interactive Diagnosis", "📊 Model Graphs & Analytics", "📚 Disease Reference"])

def show_graph(filename):
    path = f'static/graphs/{filename}' if os.path.exists(f'static/graphs/{filename}') else (f'graphs/{filename}' if os.path.exists(f'graphs/{filename}') else None)
    if path:
        try:
            st.image(path, width="stretch")
        except Exception:
            st.image(path, use_container_width=True)

all_symptoms = metadata.get('symptoms', [])
symptom_labels = {s: s.replace('_', ' ').strip().title() for s in all_symptoms}
reverse_labels = {v: k for k, v in symptom_labels.items()}

if 'selected_symptoms_list' not in st.session_state:
    st.session_state.selected_symptoms_list = []

def apply_preset(keys):
    st.session_state.selected_symptoms_list = [symptom_labels[k] for k in keys if k in symptom_labels]

with tab_diag:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("Select Patient Symptoms")
        
        # Presets
        st.write("**Quick Presets:**")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        if p_col1.button("🤒 Viral Flu"):
            apply_preset(['high_fever', 'chills', 'cough', 'headache', 'fatigue'])
        if p_col2.button("🤢 Gastroenteritis"):
            apply_preset(['vomiting', 'stomach_pain', 'diarrhoea', 'nausea'])
        if p_col3.button("🌿 Skin Infection"):
            apply_preset(['itching', 'skin_rash', 'nodal_skin_eruptions'])
        if p_col4.button("🦟 Dengue"):
            apply_preset(['high_fever', 'chills', 'joint_pain', 'vomiting', 'fatigue'])

        selected_labels = st.multiselect(
            "Choose one or more symptoms from the 132 available features:",
            options=sorted(list(symptom_labels.values())),
            key="selected_symptoms_list",
            placeholder="Type or select symptoms..."
        )

        selected_ids = [reverse_labels[label] for label in selected_labels if label in reverse_labels]

        st.caption(f"Selected symptoms count: {len(selected_ids)}")
        predict_clicked = st.button("🚀 Diagnose Condition", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Diagnostic Assessment")

        if len(selected_ids) > 0:
            feature_vector = {s: (1 if s in selected_ids else 0) for s in all_symptoms}
            input_df = pd.DataFrame([feature_vector])

            probs = model.predict_proba(input_df)[0]
            classes = model.classes_
            sorted_idx = np.argsort(probs)[::-1]

            primary_disease = classes[sorted_idx[0]]
            confidence = probs[sorted_idx[0]] * 100

            info = disease_info.get(primary_disease.strip(), disease_info.get(primary_disease, {
                'description': 'Clinical assessment',
                'specialist': 'General Physician',
                'severity': 'Moderate',
                'precautions': ['Consult a doctor for precise clinical evaluation']
            }))

            st.success(f"### Primary Diagnosis: **{primary_disease}**")
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.info(f"**Description:** {info.get('description', '')}")
            st.write(f"**Recommended Specialist:** 👨‍⚕️ {info.get('specialist', 'General Physician')}")
            st.write(f"**Severity Level:** `{info.get('severity', 'Moderate')}`")

            # Differential probabilities chart
            st.write("#### Differential Diagnoses Breakdown:")
            top_k = 5
            top_diseases = [classes[i] for i in sorted_idx[:top_k]]
            top_probs = [probs[i] * 100 for i in sorted_idx[:top_k]]

            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#0f1f1a')
            ax.set_facecolor('#0f1f1a')
            ax.barh(top_diseases[::-1], top_probs[::-1], color='#10b981', edgecolor='#34d399')
            ax.set_xlabel("Confidence (%)", color='#94a3b8')
            ax.set_xlim(0, 105)
            ax.tick_params(colors='#e2e8f0', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1e2925')
            ax.spines['bottom'].set_color('#1e2925')
            for i, v in enumerate(top_probs[::-1]):
                ax.text(v + 1.2, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=9, color='#34d399')
            plt.tight_layout()
            st.pyplot(fig)

            # Precautions
            st.write("#### Recommended Precautions:")
            for p in info.get('precautions', []):
                st.markdown(f"- 🛡️ {p}")

        else:
            st.write("👈 Select one or more symptoms on the left to view the AI diagnosis.")

with tab_graphs:
    st.subheader("Required Analytical Graphs for this Project")
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### 1. Top 20 Most Predictive Symptoms")
        show_graph('feature_importance.png')
            
        st.markdown("#### 3. Algorithm Benchmark Comparison")
        show_graph('model_comparison.png')

        st.markdown("#### 5. Disease Class Balance")
        show_graph('disease_distribution.png')

    with g_col2:
        st.markdown("#### 2. Confusion Matrix Heatmap")
        show_graph('confusion_matrix.png')

        st.markdown("#### 4. Symptoms Count Distribution")
        show_graph('symptom_distribution.png')

with tab_info:
    st.subheader("Disease Reference Directory (41 Conditions)")
    q = st.text_input("Filter diseases by keyword:", "")
    for d_name, d_info in disease_info.items():
        if not q or q.lower() in d_name.lower() or q.lower() in d_info['description'].lower():
            with st.expander(f"📌 {d_name} — {d_info.get('severity', 'Moderate')}"):
                st.write(f"**Description:** {d_info.get('description', '')}")
                st.write(f"**Specialist:** 👨‍⚕️ {d_info.get('specialist', 'General Physician')}")
                st.write("**Precautions:**")
                for pre in d_info.get('precautions', []):
                    st.write(f"- {pre}")
