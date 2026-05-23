import subprocess
import sys
import streamlit as st
import pandas as pd
import os
import io

# --- CONFIGURAÇÃO DA PÁGINA WEB ---
st.set_page_config(
    page_title="ASSET | Arthroscopy Automated Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- REESTRUTURAÇÃO COMPLETA DO CSS (Tons Hospitalares/Clínicos) ---
st.markdown("""
    <style>
    .stApp { background-color: #F4F7F9; }
    
    [data-testid="stSidebar"] {
        background-color: #0A192F !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    h1 { 
        color: #0F172A; 
        font-family: 'Segoe UI', system-ui, sans-serif; 
        font-weight: 700;
        margin-bottom: 2px;
    }
    
    .medical-card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 25px;
    }
    
    .card-title {
        color: #1E3A8A;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }

    .stButton>button {
        background-color: #0284C7 !important; 
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        height: 52px !important;
        width: 100% !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover { 
        background-color: #0369A1 !important; 
        transform: translateY(-1px);
    }
    
    .stDownloadButton>button {
        background-color: #0284C7 !important; 
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        height: 40px !important;
        width: 100% !important;
    }
    .stDownloadButton>button:hover { background-color: #0369A1 !important; }

    div.excel-btn div.stDownloadButton>button {
        background-color: #10B981 !important;
        height: 50px !important;
        font-size: 15px !important;
    }
    div.excel-btn div.stDownloadButton>button:hover { background-color: #059669 !important; }

    div[data-testid="stCheckbox"] {
        background-color: #F8FAFC;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

LOG_FILE = "resultados_asset.txt"
MASTER_SCRIPT = "ASSET_MASTER_v6.py"

SUFIXOS_VIDEOS = {
    "Safety": "SAFETY",
    "Camera Dexterity": "CAMERA",
    "Instrument Dexterity": "INSTRUMENT",
    "Bi-Manual Dexterity": "BIMANUAL",
    "Flow of Procedure": "FLOW",
    "Field of View": "FOV",
    "Autonomy": "AUTONOMY"
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color: #F8FAFC; text-align: center; font-weight: 800;'>ASSET SYSTEM</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 12px;'>Arthroscopy Automated Assessment</p>", unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader(
        "📁 LOAD SURGICAL VIDEO", 
        type=["mp4", "avi", "mov", "mkv", "m4v", "mpg"]
    )
    
    st.write("---")
    if uploaded_file:
        st.success(f"CONNECTED: {uploaded_file.name}")
    else:
        st.info("No video selected")

# --- MAIN CONTENT ---
st.title("Arthroscopy Surgical Assessment")
st.markdown("<p style='color: #64748B; font-size: 15px; margin-top:-10px;'>Unified automated evaluation system (Safety, Dexterity, FOV, Flow & Quality).</p>", unsafe_allow_html=True)

# --- CARD 1: SELETOR DE MÉTRICAS ---
st.markdown('<div class="medical-card">', unsafe_allow_html=True)
st.markdown('<h4 class="card-title">📊 Select Evaluation Metrics</h4>', unsafe_allow_html=True)

metrics_options = list(SUFIXOS_VIDEOS.keys()) + ["Quality of Procedure"]

col1, col2, col3, col4 = st.columns(4)
selected_metrics = []

for i, metric in enumerate(metrics_options):
    with [col1, col2, col3, col4][i % 4]:
        if st.checkbox(metric, value=True, key=metric):
            selected_metrics.append(metric)

st.markdown('</div>', unsafe_allow_html=True)

# --- CARD 2: PAINEL DE EXECUÇÃO ---
st.markdown('<div class="medical-card">', unsafe_allow_html=True)
st.markdown('<h4 class="card-title">🩺 Clinical Diagnostic Engine</h4>', unsafe_allow_html=True)

col_btn, _ = st.columns([2, 3])
with col_btn:
    execution_clicked = st.button("START AUTOMATED EVALUATION")

if execution_clicked:
    if not uploaded_file:
        st.error("Medical Alert: No surgical video has been loaded!")
    elif not selected_metrics:
        st.error("Selection Error: Please select at least one metric!")
    elif not os.path.exists(MASTER_SCRIPT):
        st.error(f"System Error: Script missing ({MASTER_SCRIPT})")
    else:
        log_absoluto = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
        if os.path.exists(log_absoluto):
            os.remove(log_absoluto)
            
        status_warning = st.warning("🔬 Pipeline Active: Running Computer Vision models. Please wait...")
        
        try:
            # --- CORREÇÃO DE PREFIXO 'temp_' ---
            nome_original = uploaded_file.name
            if nome_original.startswith("temp_"):
                nome_original = nome_original.replace("temp_", "", 1)
            
            temp_video_path = nome_original
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Prepara os argumentos com caminhos absolutos baseados no executável global ativo
            args = [sys.executable, MASTER_SCRIPT, temp_video_path] + selected_metrics
            pasta_atual = os.path.dirname(os.path.abspath(__file__))
            
            # --- CONFIGURAÇÃO CIRÚRGICA DE VARIÁVEIS DE AMBIENTE ---
            env_atual = os.environ.copy()
            env_atual["QT_QPA_PLATFORM"] = "offscreen"  # Bloqueia chamadas visuais em servidores headless
            env_atual["OPENCV_LOG_LEVEL"] = "OFF"
            if "DISPLAY" not in env_atual:
                env_atual["DISPLAY"] = ":0"
            
            # Executa capturando explicitamente o stdout e o stderr de forma separada
            resultado_processo = subprocess.run(
                args, 
                cwd=pasta_atual, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                env=env_atual
            )
            
            status_warning.empty()
            
            # --- INTERCEÇÃO E EXIBIÇÃO EXPLICITA DE ERROS DO PIPELINE ---
            if resultado_processo.returncode != 0:
                st.error(f"❌ O pipeline de IA falhou internamente! (Código de erro técnico: {resultado_processo.returncode})")
                
                # Exibe o rastreio do erro real do Python (Stderr)
                st.markdown("### 🛑 Traceback / Erro Detalhado do Motor:")
                if resultado_processo.stderr:
                    st.code(resultado_processo.stderr, language="python")
                else:
                    st.warning("Nenhuma informação enviada para o canal Stderr.")
                
                # Exibe logs de print comuns disparados até ao momento do crash (Stdout)
                st.markdown("### 📋 Fluxo de Saída Interrompido (Stdout):")
                if resultado_processo.stdout:
                    st.code(resultado_processo.stdout, language="text")
                else:
                    st.warning("Nenhuma informação enviada para o canal Stdout.")
                
                # Interrompe a execução para não quebrar a UI com tabelas vazias
                st.stop()
            
            # Se correu perfeitamente (returncode == 0)
            else:
                if not os.path.exists(log_absoluto):
                    st.error("Data Error: No results file was generated.")
                else:
                    st.success("Analysis Completed Successfully!")
                    
                    resultados = []
                    with open(log_absoluto, "r") as f:
                        for linha in f:
                            if ":" in linha:
                                partes = linha.strip().split(":")
                                if len(partes) < 2: 
                                    continue
                                metrica = partes[0].strip()
                                try:
                                    nota_val = float(partes[1])
                                    if metrica in selected_metrics:
                                        resultados.append({
                                            "Metric": metrica,
                                            "Score (1-5)": int(nota_val),
                                            "Source Video": uploaded_file.name
                                        })
                                if ValueError:
                                    continue
                    
                    df = pd.DataFrame(resultados).drop_duplicates(subset=['Metric'], keep='last')
                    
                    if df.empty:
                        st.warning("No matching metrics data found.")
                    else:
                        st.subheader("Assessment Scores")
                        st.dataframe(df, use_container_width=True)
                        
                        nome_base = temp_video_path.split('.')[0]
                        nome_aluno = nome_base.replace('_RAFT_ANALISE', '')
                        pasta_saida_servidor = f"Avaliações_VID_{nome_aluno}"
                        
                        st.write("---")
                        st.subheader("📥 Export Clinical Reports & Processed Videos")
                        
                        st.markdown('<div class="excel-btn">', unsafe_allow_html=True)
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Surgical_Assessment')
                        
                        st.download_button(
                            label="📊 DOWNLOAD EXCEL CLINICAL REPORT",
                            data=buffer.getvalue(),
                            file_name=f"REPORT_ASSET_{nome_aluno}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write("")
                        
                        if os.path.exists(pasta_saida_servidor):
                            col_v1, col_v2, col_v3 = st.columns(3)
                            index_col = 0
                            
                            for m_nome, sufixo in SUFIXOS_VIDEOS.items():
                                nome_video_esperado = f"{nome_aluno}_{sufixo}.mp4"
                                caminho_completo_video = os.path.join(pasta_saida_servidor, nome_video_esperado)
                                
                                if os.path.exists(caminho_completo_video):
                                    alvo_col = [col_v1, col_v2, col_v3][index_col % 3]
                                    with alvo_col:
                                        with open(caminho_completo_video, "rb") as vf:
                                            st.download_button(
                                                label=f"🎬 {m_nome}",
                                                data=vf.read(),
                                                file_name=nome_video_esperado,
                                                mime="video/mp4",
                                                key=f"dl_{sufixo}"
                                            )
                                    index_col += 1
                            
                            if index_col == 0:
                                st.info("No processed videos were found in the output folder.")
                        else:
                            st.error(f"Output folder not found: {pasta_saida_servidor}")
                            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                
        except Exception as e:
            status_warning.empty()
            st.error(f"System Failure during execution: {e}")

if not execution_clicked:
    st.info("ℹ️ System Status: Idle. Load a video file via sidebar to begin diagnostic pipeline.")

st.markdown('</div>', unsafe_allow_html=True)