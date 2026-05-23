# -*- coding: utf-8 -*-
"""
ASSET PROCESSOR - WEB & LOCAL UNIFIED (ON-THE-FLY ENGINE)
"""
import sys
import os

# --- 0. CONFIGURAÇÃO ANTI-CRASH PARA AMBIENTES HEADLESS (NUVEM) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# Se o Linux do Streamlit tentar forçar o carregamento do módulo gráfico nativo,
# nós injetamos um "mock" no sistema de módulos para o Python ignorar o linker do OS.
try:
    import cv2
except ImportError as e:
    if "libGL.so.1" in str(e):
        # Desativa a validação gráfica do módulo nativo
        sys.modules['cv2.cv2'] = None
        # Força o fallback para o backend matemático puro
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        import cv2
    else:
        raise e

import cv2
import numpy as np
import pandas as pd
import math
import torch
from ultralytics import YOLO
from collections import deque

# --- 1. CONFIGURAÇÕES E DIRETÓRIOS ---


# --- 1. CONFIGURAÇÕES E DIRETÓRIOS ---
PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
MODELO_YOLO_PATH = os.path.join(PASTA_RAIZ, "best.pt")

if not os.path.exists(MODELO_YOLO_PATH) and os.path.exists("best.pt"):
    MODELO_YOLO_PATH = "best.pt"

PASTA_MOVIMENTO = os.path.join(PASTA_RAIZ, "Dados_Movimento")
PASTA_MAE_SAIDA = PASTA_RAIZ
os.makedirs(PASTA_MOVIMENTO, exist_ok=True)

# Constantes Cirúrgicas
AREA_MAXIMA = 45000
COOLDOWN_EVENTO = 3.0
LIMIAR_TOQUE = 5.0
LIMIAR_DANOS = 10.0
LIMIAR_BRUSCO_RAFT = 8.0
SEGUNDOS_ESPERA_INICIAL = 5.0
SEGUNDOS_MEMORIA = 3.0
LIMIAR_DIF_ESTATICO = 1.2   
TEMPO_LIMITE_ESTATICO = 7.0 
TEMPO_LIMITE_AUSENCIA = 10.0 

# --- 2. CONFIGURAÇÃO DO MODELO DE PROFUNDIDADE (MiDaS On-The-Fly) ---
print("🔬 A carregar modelo de profundidade inteligente MiDaS (Small)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform

# --- 3. SETUP DE ENTRADA E ARGUMENTOS ---
if len(sys.argv) > 2:
    VIDEO_PATH = sys.argv[1]
    METRICAS_SELECIONADAS = sys.argv[2:]
else:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    VIDEO_PATH = filedialog.askopenfilename(title="Selecionar Vídeo para Avaliação ASSET")
    if not VIDEO_PATH: sys.exit()
    METRICAS_SELECIONADAS = ["Safety", "Field of View", "Camera Dexterity", "Instrument Dexterity", 
                             "Bi-Manual Dexterity", "Flow of Procedure", "Quality of Procedure", "Autonomy"]

nome_base = os.path.basename(VIDEO_PATH).split('.')[0]
if nome_base.startswith("temp_"):
    nome_base = nome_base.replace("temp_", "", 1)

NOME_ALUNO = nome_base.replace('_RAFT_ANALISE', '')
path_csv = os.path.join(PASTA_MOVIMENTO, f"{NOME_ALUNO}_dados_flow.csv")

yolo_model = YOLO(MODELO_YOLO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

W_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
escala_x, escala_y = 1.0, 1.0
RAIO_FOV_CENTRO = int(H_ORIG * 0.15)

# --- 4. CALIBRAÇÃO DINÂMICA DO CENTRO ---
cap.set(cv2.CAP_PROP_POS_FRAMES, min(2000, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-5)) 
ret, frame_ref = cap.read()
centro_calibrado = (W_ORIG / 2, H_ORIG / 2) 

if ret:
    gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    blurred_ref = cv2.medianBlur(gray_ref, 7)
    min_r = int(H_ORIG * 0.35)  
    max_r = int(H_ORIG * 0.55)  
    circles = cv2.HoughCircles(blurred_ref, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150, 
                               param1=50, param2=35, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circles = np.float32(np.around(circles))
        centro_calibrado = (circles[0, 0][0], circles[0, 0][1])

cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset total para processar desde o início

# --- 5. SETUP DE SAÍDA (WRITERS SELETIVOS) ---
pasta_final = os.path.join(PASTA_MAE_SAIDA, f"Avaliações_VID_{NOME_ALUNO}")
os.makedirs(pasta_final, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def criar_writer(nome_metrica, sufixo):
    if nome_metrica in METRICAS_SELECIONADAS:
        return cv2.VideoWriter(os.path.join(pasta_final, f"{NOME_ALUNO}_{sufixo}.mp4"), fourcc, fps, (W_ORIG, H_ORIG))
    return None

video_safety_out = criar_writer("Safety", "SAFETY")
video_camera_out = criar_writer("Camera Dexterity", "CAMERA")
video_instr_out = criar_writer("Instrument Dexterity", "INSTRUMENT")
video_bimanual_out = criar_writer("Bi-Manual Dexterity", "BIMANUAL")
video_flow_out = criar_writer("Flow of Procedure", "FLOW")
video_fov_out = criar_writer("Field of View", "FOV")
video_autonomia_out = criar_writer("Autonomy", "AUTONOMY")

# --- 6. ACUMULADORES E AUXILIARES ---
def calcular_jitter(valores):
    if len(valores) < 2: return 0.0
    return np.mean(np.abs(np.diff(valores)))

def atribuir_nota_jitter(v):
    return 5 if v <= 0.45 else 4 if v <= 0.65 else 3 if v <= 0.85 else 2 if v <= 1.2 else 1

def atribuir_nota_raft(v):
    return 5 if v <= 0.22 else 4 if v <= 0.38 else 3 if v <= 0.55 else 2 if v <= 0.75 else 1

def desenhar_painel_hud(img, lista_textos, cores):
    FONTE = cv2.FONT_HERSHEY_SIMPLEX
    ESC, ESP, ALT_LINHA, MARGEM_X, MARGEM_Y = 0.55, 1, 25, 20, 30
    largura_painel, altura_painel = 300, (len(lista_textos) * ALT_LINHA) + 20
    cv2.rectangle(img, (10, 10), (largura_painel, altura_painel), (0, 0, 0), -1)
    for i, (texto, cor) in enumerate(zip(lista_textos, cores)):
        pos_y = MARGEM_Y + (i * ALT_LINHA)
        cv2.putText(img, texto, (MARGEM_X, pos_y), FONTE, ESC, cor, ESP, cv2.LINE_AA)

c      
cont_danos, cont_toque, cont_falhas_fluxo, cont_bruscos_fluxo = 0, 0, 0, 0
ultimo_evento_reg, ultima_falha_reg, frame_ultimo_brusco = -int(fps*COOLDOWN_EVENTO), -int(fps), -999
frames_com_yolo, frames_com_objeto_quality, frames_vistos_probe, frames_em_contato_alvo = 0, 0, 0, 0
historico_jitter_instr, historico_jitter_camera, historico_fov_dist, historico_centro_fov = [], [], [], []
historico_ang_cam = []
historico_brilho_bruto = []
historico_probe_recent = deque(maxlen=int(fps * SEGUNDOS_MEMORIA))
pos_ant_probe = None
estado_bimanual = 1
contador_ausencia_ferramenta = 0
dados_bimanual = {1: {"instr": [], "raft": []}, 2: {"instr": [], "raft": []}}
SEGUNDOS_MEMORIA_SAFETY = 5.0 
historico_probe_recent = deque(maxlen=int(fps * SEGUNDOS_MEMORIA_SAFETY))
contato_triangulo_ocorrido = False 
alvo_ja_foi_visto = False
contador_ausencia_ferramenta = 0
LIMIAR_AUSENCIA = int(fps * 5) # 5 segundos de ausência para trocar de mão
estado_bimanual = 1
fase_transicao = "AGUARDANDO_INICIO" # Fases: AGUARDANDO_INICIO, PROCURANDO_TRIANGULO, CONTANDO_SAIDA
contador_ausencia_total = 0
frames_parado_consecutivos = 0
contador_intervencoes = 0
buffer_mov_autonomia = deque(maxlen=15) # Analisa meio segundo de vídeo para decidir se parou
frames_ausencia_consecutivos = 0
frames_estaticos_consecutivos = 0



# Lista para acumular os valores calculados do RAFT e guardar no arquivo CSV final
valores_calculados_raft = []

ret, f_init = cap.read()
if not ret: sys.exit("Erro: Ficheiro de vídeo danificado ou vazio.")
prev_gray = cv2.GaussianBlur(cv2.cvtColor(f_init, cv2.COLOR_BGR2GRAY), (7,7), 0)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0

# --- 7. LOOP DE PROCESSAMENTO PRINCIPAL ---
while cap.isOpened():
    ret, frame_raw = cap.read()
    if not ret: break

    # Variáveis anti-crash
    brilho, dist_c, erro_horizonte, jitter_inst_atual = 0.0, 0.0, 0.0, 0.0
    validar_contato, status_flow = False, "NAVEGACAO"
    pt_orig = (0, 0)
    instabilidade_txt = "Instabilidade camara: -"
    horizontalidade_txt = "Horizontalidade: -"
    dist_centro_camara_txt = "Distancia ao centro: -"
    
    frame_proc = frame_raw.copy()
    frame_draw = frame_raw.copy()
    gray_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_proc, (7,7), 0)

    # A. CÁLCULO DO OPTICAL FLOW (NATIVO & FRAME-A-FRAME)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_blur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude_raft_frame = float(np.mean(mag))
    valores_calculados_raft.append(magnitude_raft_frame)
    historico_jitter_camera.append(magnitude_raft_frame)

    # Detetar movimentos bruscos com base no cálculo em tempo real
    if magnitude_raft_frame > LIMIAR_BRUSCO_RAFT:
        if frame_idx - frame_ultimo_brusco > 60:
            cont_bruscos_fluxo += 1
            frame_ultimo_brusco = frame_idx

    mag[mag < 0.8] = 0
    mov_global = np.median(mag)

    # B. ESTIMATIVA DE PROFUNDIDADE (ON-THE-FLY VIA MIDAS)
    img_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    # Normalização para escala cinzenta (0-255)
    depth_gray = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # C. DETEÇÃO YOLO
    results = yolo_model.predict(frame_proc, conf=0.35, verbose=False)
    mask_probe = np.zeros((H_ORIG, W_ORIG), dtype=np.uint8)
    mask_alvos = np.zeros((H_ORIG, W_ORIG), dtype=np.uint8)
    alvos_atuais = {}
    centro_probe_atual, probe_no_frame = None, False

    if results[0].masks is not None:
        frames_com_yolo += 1
        for mask_data, box in zip(results[0].masks.xy, results[0].boxes):
            pts = mask_data.astype(np.int32)
            if cv2.contourArea(pts) > AREA_MAXIMA: continue
            label = yolo_model.names[int(box.cls)].lower()
            
            if "probe" in label:
                probe_no_frame = True
                frames_vistos_probe += 1
                cv2.fillPoly(mask_probe, [pts], 255)
                M = cv2.moments(pts)
                if M["m00"] != 0: centro_probe_atual = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            elif any(t in label for t in ["green_screw", "red_triangle", "red_cross"]):
                frames_com_objeto_quality += 1
                cv2.fillPoly(mask_alvos, [pts], 255)
                x1, y1, x2, y2 = box.xyxy[0]
                alvos_atuais[label] = ((x1+x2)/2, (y1+y2)/2)
                roi = gray_proc[max(0,int(y1)):min(H_ORIG,int(y2)), max(0,int(x1)):min(W_ORIG,int(x2))]
                if roi.size > 0:
                    brilho = float(np.mean(roi))
                    historico_brilho_bruto.append(brilho)

    historico_probe_recent.append(probe_no_frame)

    # D. AUTONOMIA & SEGURANÇA
    nomes_alvos = ["green_screw", "red_triangle", "red_cross"]
    alvo_visivel = any(name in alvos_atuais for name in nomes_alvos)
    presenca_detetada = probe_no_frame or alvo_visivel
    
    diff_frames = cv2.absdiff(prev_gray, gray_blur)
    imagem_parada = np.mean(diff_frames) < LIMIAR_DIF_ESTATICO

    if not presenca_detetada:
        frames_ausencia_consecutivos += 1
    else:
        if (frames_ausencia_consecutivos / fps) >= TEMPO_LIMITE_AUSENCIA:
            contador_intervencoes += 1
        frames_ausencia_consecutivos = 0

    if imagem_parada:
        frames_estaticos_consecutivos += 1
    else:
        if (frames_estaticos_consecutivos / fps) >= TEMPO_LIMITE_ESTATICO:
            contador_intervencoes += 1
        frames_estaticos_consecutivos = 0
    
    mask_analise = np.zeros((H_ORIG, W_ORIG), dtype=np.uint8)
    cv2.circle(mask_analise, (int(centro_calibrado[0]), int(centro_calibrado[1])), int(H_ORIG//2.5), 255, -1)
    mask_alvos_dilatada = cv2.dilate(mask_alvos, np.ones((10,10), np.uint8))
    mask_tecido_limpa = cv2.bitwise_and(mask_analise, cv2.bitwise_not(mask_alvos_dilatada))
    mask_tecido_limpa = cv2.bitwise_and(mask_tecido_limpa, cv2.bitwise_not(mask_probe))
    
    mags_tecido = mag[mask_tecido_limpa > 0]
    mov_local = np.percentile(mags_tecido, 95) if mags_tecido.size > 0 else 0
    probe_recente_ativa = any(historico_probe_recent)
    contato_com_alvo = np.any(cv2.bitwise_and(mask_probe, mask_alvos))

    if (frame_idx / fps) >= SEGUNDOS_ESPERA_INICIAL and probe_recente_ativa and not contato_com_alvo:
        if mov_local > (mov_global + LIMIAR_DANOS) or mov_local > (mov_global + LIMIAR_TOQUE):
            if frame_idx - ultimo_evento_reg > (fps * COOLDOWN_EVENTO):
                if mov_local > (mov_global + LIMIAR_DANOS): cont_danos += 1
                else: cont_toque += 1
                ultimo_evento_reg = frame_idx

    # E. FLUXO, ORIENTAÇÃO E CONTATO BRUSCO
    if probe_no_frame and alvos_atuais:
        M_p, M_a = cv2.moments(mask_probe), cv2.moments(mask_alvos)
        if M_p["m00"] != 0 and M_a["m00"] != 0:
            cX_p, cY_p = int(M_p["m10"] / M_p["m00"]), int(M_p["m01"] / M_p["m00"])
            cX_a, cY_a = int(M_a["m10"] / M_a["m00"]), int(M_a["m01"] / M_a["m00"])
            distancia_centros = math.sqrt((cX_p - cX_a)**2 + (cY_p - cY_a)**2)

            std_p = np.std(depth_gray[mask_probe == 255])
            std_a = np.std(depth_gray[mask_alvos == 255])
            intersecao = cv2.bitwise_and(mask_probe, mask_alvos)

            if np.any(intersecao) or (abs(std_p - std_a) < 4.0 and distancia_centros <= 15.0):
                validar_contato = True
                status_flow = "CONTACTO"
                frames_em_contato_alvo += 1
            else:
                status_flow = "FALHA"
                if frame_idx - ultima_falha_reg > 3 * fps:
                    cont_falhas_fluxo += 1
                    ultima_falha_reg = frame_idx

    # F. DESTREZA DA CÂMARA
    c_x_orig, c_y_orig = int(centro_calibrado[0]), int(centro_calibrado[1])
    if alvos_atuais:
        label, pt = list(alvos_atuais.items())[0]
        pt_orig = (int(pt[0]), int(pt[1]))
        dist_c = math.sqrt((pt[0]-centro_calibrado[0])**2 + (pt[1]-centro_calibrado[1])**2)
        historico_fov_dist.append(dist_c)
        
        dx, dy = pt[0]-centro_calibrado[0], -(pt[1]-centro_calibrado[1])
        ang = float(np.degrees(np.arctan2(float(dy), float(dx)))) % 360
        ref = min([0, 90, 180, 270], key=lambda x: min(abs(x-ang), 360-abs(x-ang)))
        erro_horizonte = min(abs(ang-ref), 360-abs(ang-ref))
        historico_ang_cam.append(erro_horizonte)
        
    instabilidade_txt = f"Instabilidade camara: {magnitude_raft_frame:.2f}"
    horizontalidade_txt = f"Horizontalidade: {erro_horizonte:.1f}"
    dist_centro_camara_txt = f"Distancia ao centro: {dist_c:.1f}px"

    # G. DESTREZA DO INSTRUMENTO & ESTADOS BIMANUAL
    if centro_probe_atual and pos_ant_probe:
        dist_p = math.sqrt((centro_probe_atual[0]-pos_ant_probe[0])**2 + (centro_probe_atual[1]-pos_ant_probe[1])**2)
        if 0.5 < dist_p < 80:
            jitter_inst_atual = dist_p
            historico_jitter_instr.append(jitter_inst_atual)
    
    perc_direcionamento = (frames_em_contato_alvo / frames_vistos_probe * 100) if frames_vistos_probe > 0 else 0.0

    tem_green_screw = any("green_screw" in label for label in alvos_atuais.keys())
    tem_red_triangle = any("red_triangle" in label for label in alvos_atuais.keys())
    tem_ambos_saida = (tem_red_triangle and probe_no_frame)
    
    if estado_bimanual == 1:
        if fase_transicao == "AGUARDANDO_INICIO" and tem_green_screw:
            fase_transicao = "PROCURANDO_TRIANGULO"
        elif fase_transicao == "PROCURANDO_TRIANGULO" and tem_ambos_saida:
            fase_transicao = "CONTANDO_SAIDA"
        elif fase_transicao == "CONTANDO_SAIDA":
            if not probe_no_frame and not tem_red_triangle:
                contador_ausencia_total += 1
            else: contador_ausencia_total = 0 
            if contador_ausencia_total > (fps * 5):
                estado_bimanual = 2
                fase_transicao = "FINALIZADO"

    dados_bimanual[estado_bimanual]["instr"].append(jitter_inst_atual)
    dados_bimanual[estado_bimanual]["raft"].append(magnitude_raft_frame)

    tempo_estatico = frames_estaticos_consecutivos / fps
    tempo_ausencia = frames_ausencia_consecutivos / fps
    cor_e = (0, 0, 255) if tempo_estatico > TEMPO_LIMITE_ESTATICO else (255, 255, 255)
    cor_a = (0, 0, 255) if tempo_ausencia > TEMPO_LIMITE_AUSENCIA else (255, 255, 255)

    # --- H. RENDERIZAÇÃO DE HUD SELETIVA ---
    if video_safety_out is not None:
        frame_s = frame_draw.copy()
        desenhar_painel_hud(frame_s, [f"DANOS: {cont_danos}", f"TOQUES: {cont_toque}"], [(0, 0, 255), (0, 165, 255)])
        video_safety_out.write(frame_s)

    if video_flow_out is not None:
        frame_f = frame_draw.copy()
        status_cor = (0, 255, 0) if status_flow == "CONTACTO" else (0, 0, 255) if status_flow == "FALHA" else (255, 255, 255)
        cor_b = (0, 165, 255) if (frame_idx - frame_ultimo_brusco < 30) else (255, 255, 255)
        desenhar_painel_hud(frame_f, [f"Estado: {status_flow}", f"Falhas relativas ao alvo: {cont_falhas_fluxo}", f"Mov. Bruscos: {cont_bruscos_fluxo}"], [status_cor, (255, 255, 255), cor_b])
        video_flow_out.write(frame_f)

    if video_camera_out is not None:
        frame_c = frame_draw.copy()
        desenhar_painel_hud(frame_c, [instabilidade_txt, horizontalidade_txt, dist_centro_camara_txt], [(0, 255, 255), (255, 255, 0), (0, 255, 0)])
        if alvos_atuais: cv2.line(frame_c, (c_x_orig, c_y_orig), pt_orig, (255, 0, 0), 2)
        cv2.drawMarker(frame_c, (c_x_orig, c_y_orig), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame_c, (c_x_orig, c_y_orig), int(RAIO_FOV_CENTRO), (255, 255, 255), 1)
        video_camera_out.write(frame_c)

    if video_fov_out is not None:
        frame_fov = frame_draw.copy()
        desenhar_painel_hud(frame_fov, [f"Luz: {brilho:.1f}", f"Distancia ao centro: {dist_c:.1f}px"], [(0, 255, 0), (0, 255, 255)])
        cv2.drawMarker(frame_fov, (c_x_orig, c_y_orig), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        video_fov_out.write(frame_fov)

    if video_instr_out is not None:
        frame_i = frame_draw.copy()
        status_instr = "CONTACTO" if validar_contato else "NAVEGACAO"
        cor_status = (0, 255, 0) if validar_contato else (0, 165, 255)
        desenhar_painel_hud(frame_i, [f"Instabilidade Instrumento: {jitter_inst_atual:.2f}", f"Corretamente Orientada: {perc_direcionamento:.1f}%", f"Estado: {status_instr}"], [(255, 255, 255), (255, 255, 255), cor_status])
        video_instr_out.write(frame_i)

    if video_bimanual_out is not None:
        frame_b = frame_raw.copy()
        mao_nome = "DIREITA" if estado_bimanual == 1 else "ESQUERDA"
        cor_mao = (255, 128, 0) if estado_bimanual == 1 else (255, 0, 255)
        desenhar_painel_hud(frame_b, [f"MAO: {mao_nome}", f"Instabilidade Instrumento: {jitter_inst_atual:.2f}", f"Instabilidade Camara: {magnitude_raft_frame:.2f}"], [cor_mao, (200, 200, 200), (255, 255, 255)])
        video_bimanual_out.write(frame_b)

    if video_autonomia_out is not None:
        frame_a = frame_draw.copy()
        desenhar_painel_hud(frame_a, [f"INTERVENCOES: {contador_intervencoes}", f"Camara Estatica: {tempo_estatico:.1f}s / {TEMPO_LIMITE_ESTATICO}s", f"Ausencia de alvos: {tempo_ausencia:.1f}s / {TEMPO_LIMITE_AUSENCIA}s"], [(255, 255, 255), cor_e, cor_a])
        video_autonomia_out.write(frame_a)

    if centro_probe_atual: pos_ant_probe = centro_probe_atual
    prev_gray = gray_blur.copy()
    frame_idx += 1

# --- 8. CÁLCULO E LOGICA DE EXPORTAÇÃO (CSV E METRICAS) ---
cap.release()

# EXPORTAR CSV DE MOVIMENTO (PEDIDO EXPLICITO)
df_novo_flow = pd.DataFrame({"magnitude_raft": valores_calculados_raft})
df_novo_flow.to_csv(path_csv, index=False)
print(f"✅ Ficheiro CSV de movimento gerado com sucesso em: {path_csv}")
 
# --- 6. CÁLCULO DAS NOTAS FINAIS ---
if frame_idx == 0: sys.exit("Erro: Vídeo sem frames processados.")

# [SAFETY]
if (cont_danos <= 1 and cont_toque <= 3): n_safety = 5
elif cont_danos <= 3 and cont_toque <= 7: n_safety = 4
elif cont_danos <= 5 and cont_toque <= 8: n_safety = 3
elif cont_danos <= 8 and cont_toque <= 10: n_safety = 2
else: n_safety = 1

# [BIMANUAL]
j_dir = np.mean(np.abs(np.diff(dados_bimanual[1]["instr"]))) if len(dados_bimanual[1]["instr"]) > 1 else 0
r_dir = np.mean(dados_bimanual[1]["raft"]) if dados_bimanual[1]["raft"] else 0
j_esq = np.mean(np.abs(np.diff(dados_bimanual[2]["instr"]))) if len(dados_bimanual[2]["instr"]) > 1 else 0
r_esq = np.mean(dados_bimanual[2]["raft"]) if dados_bimanual[2]["raft"] else 0

# [BIMANUAL] - Versão Protegida
j_dir_val = atribuir_nota_jitter(j_dir) if j_dir > 0 else 0
r_dir_val = atribuir_nota_raft(r_dir) if r_dir > 0 else 0
j_esq_val = atribuir_nota_jitter(j_esq) if j_esq > 0 else 0
r_esq_val = atribuir_nota_raft(r_esq) if r_esq > 0 else 0

# Faz a média apenas das notas que não são zero
notas_validas = [v for v in [j_dir_val, r_dir_val, j_esq_val, r_esq_val] if v > 0]
n_mov_global = np.mean(notas_validas) if notas_validas else 1

p_yolo = (frames_com_objeto_quality / frame_idx) * 100
n_vis = 5 if p_yolo >= 75 else 4 if p_yolo >= 65 else 3 if p_yolo >= 60 else 2 if p_yolo >= 55 else 1
n_bim = int(round((n_mov_global + n_vis) / 2))

# [CAMERA DEXTERITY]
h_med = np.mean(historico_ang_cam) if historico_ang_cam else 99
f_med = np.mean(historico_fov_dist) if historico_fov_dist else 999
j_med_cam = np.mean(historico_jitter_camera) if historico_jitter_camera else 99

if p_yolo >= 75 and h_med <= 25 and f_med <= 130 and j_med_cam <= 0.15: n_camera = 5
elif p_yolo >= 65 and h_med <= 29 and f_med <= 140 and j_med_cam <= 0.20: n_camera = 4
elif p_yolo >= 60 and h_med <= 30 and f_med <= 150 and j_med_cam <= 0.25: n_camera = 3 
elif p_yolo >= 55 and h_med <= 30 and f_med <= 160 and j_med_cam <= 0.3: n_camera = 2
else: n_camera = 1

# [INSTRUMENT DEXTERITY]
hesitacao = np.mean(historico_jitter_instr) + (np.std(historico_jitter_instr) * 0.5) if historico_jitter_instr else 99
perc_contato = (frames_em_contato_alvo / frames_vistos_probe * 100) if frames_vistos_probe > 0 else 0
n_f = 5 if hesitacao <= 9 else 4 if hesitacao <= 10 else 3 if hesitacao < 11 else 2 if hesitacao < 13 else 1
n_d = 5 if perc_contato >= 70 else 4 if perc_contato >= 60 else 3 if perc_contato >= 50 else 2 if perc_contato >= 40 else 1
n_instr = int(round((n_f * 0.5) + (n_d * 0.5)))

# [FIELD OF VIEW]
m_c = np.mean(historico_fov_dist) if historico_fov_dist else 0
m_l = np.mean(historico_brilho_bruto) if historico_brilho_bruto else 0
# Na artroscopia, o brilho ideal situa-se entre 90 e 150. A distância ideal é dentro do raio central.
if p_yolo >= 75 and m_c <= 130 and m_l >= 78: 
    n_fov = 5
elif p_yolo >= 65 and m_c <= 140 and m_l >= 70: 
    n_fov = 4
elif p_yolo >= 60 and m_c <= 150 and m_l >= 60: 
    n_fov = 3
elif p_yolo >= 55 and m_c <= 160 and m_l >= 50: 
    n_fov = 2
else: 
    n_fov = 1


# [FLOW OF PROCEDURE]
if cont_falhas_fluxo <= 12 and cont_bruscos_fluxo <= 4 and p_yolo >= 75 and frame_idx <= 4700: n_flow = 5
elif cont_falhas_fluxo <= 15 and cont_bruscos_fluxo <= 5 and p_yolo >= 65 and frame_idx <= 8000: n_flow = 4
elif cont_falhas_fluxo <= 25 and cont_bruscos_fluxo <= 6 and p_yolo >= 60 and frame_idx <= 9000: n_flow = 3
elif cont_falhas_fluxo <= 30 and cont_bruscos_fluxo <= 7 and p_yolo >= 55 and frame_idx <= 15000: n_flow = 3
else: n_flow = 1

# [QUALITY OF PROCEDURE]
n_visao = 5 if p_yolo >= 75 else 4 if p_yolo >= 65 else 3 if p_yolo >= 60 else 2 if p_yolo >= 55 else 1
n_tempo = 5 if frame_idx <= 4700 else 4 if  frame_idx <= 8000 else 3 if frame_idx <= 9000 else 2 if frame_idx <= 15000 else 1
#n_quality = int(round((n_visao + n_tempo) / 2))


# [AUTONOMIA]
n_autonomia = 1

if contador_intervencoes <= 6:
    n_autonomia = 3
elif contador_intervencoes <= 9:
    n_autonomia = 2
else: n_autonomia = 1
    
    
    
# 2. Conversão da Autonomia para escala de 5 (apenas para cálculo interno)
    # Mapeamento: 3->5, 2->3, 1->1
n_autonomia_para_media = 5 if n_autonomia == 3 else 3 if n_autonomia == 2 else 1   

# 3. Cálculo da Média Global (usando a autonomia convertida)
notas_para_global = [n_safety, n_fov, n_camera, n_instr, n_bim, n_flow, n_autonomia_para_media]
media_global_outros = sum(notas_para_global) / len(notas_para_global) 

fator_performance = (n_visao + n_tempo) / 2

# 5. Nota Final de Qualidade
# Média entre a (Média de tudo o resto com autonomia convertida) + (Fatores de Qualidade)
n_quality = int(round((media_global_outros + fator_performance) / 2))


# --- 7. EXPORTAÇÃO ---
# No final do script, antes de fechar os arquivos:

# --- 7. EXPORTAÇÃO FINAL ---
res_completo = {
    "Safety": n_safety, "Field of View": n_fov, "Camera Dexterity": n_camera,
    "Instrument Dexterity": n_instr, "Bi-Manual Dexterity": n_bim,
    "Flow of Procedure": n_flow, "Quality of Procedure": n_quality,
    "Autonomy": n_autonomia
}

# Caminho absoluto para evitar que a UI não o encontre
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_asset.txt")

with open(log_path, "w") as f:
    for k, v in res_completo.items():
        # IMPORTANTE: Escrevemos sempre todas as métricas no TXT para a UI poder calcular a média,
        # mas a UI só mostrará no Excel as que o utilizador selecionou.
        f.write(f"{k}:{v}\n")

print(f"Ficheiro de resultados gerado em: {log_path}")

# Guardar Resultados para a Interface Streamlit
res_completo = {
    "Safety": n_safety, "Field of View": n_fov, "Camera Dexterity": n_camera,
    "Instrument Dexterity": n_instr, "Bi-Manual Dexterity": n_bim,
    "Flow of Procedure": n_flow, "Quality of Procedure": n_quality, "Autonomy": n_autonomia
}

log_path = os.path.join(PASTA_RAIZ, "resultados_asset.txt")
with open(log_path, "w") as f:
    for k, v in res_completo.items(): f.write(f"{k}:{v}\n")

# Fechar os VideoWriters abertos
for writer in [video_safety_out, video_camera_out, video_instr_out, video_bimanual_out, video_flow_out, video_fov_out, video_autonomia_out]:
    if writer: writer.release()
cv2.destroyAllWindows()
print("🎉 Avaliação terminada com sucesso e ficheiros de vídeo guardados.")




print("\n" + "="*60)
print(f" RELATÓRIO TÉCNICO INTEGRADO ASSET: {NOME_ALUNO}")
print("="*60)
print(f" [SAFETY] Danos: {cont_danos} | Toques: {cont_toque} -> Nota: {n_safety}")
print(f" [FOV] Distância ao centro: {m_c:.2f} | Iluminação: {m_l:.2f} -> Nota: {n_fov}")
print(f" [CAMARA] Erro horizonte: {h_med:.2f} | Intsbailidade da camara: {j_med_cam:.2f} -> Nota: {n_camera}")
print(f" [INSTRUMENT] Instabilidade do instrumento: {hesitacao:.2f} | Frames onde o instrumento está corretamente orientado: {perc_contato:.1f}% -> Nota: {n_instr}")
print(f" [BIMANUAL] Instabilidade instrumento D/E: {j_dir:.2f}/{j_esq:.2f}")
print(f"            Instabilidade camara D/E:   {r_dir:.2f}/{r_esq:.2f} -> Nota: {n_bim}")
print(f" [FLOW] Contagem de Falhas relativas ao alvo: {cont_falhas_fluxo} | Contagem de Mov bruscos: {cont_bruscos_fluxo} -> Nota: {n_flow}")
print(f" [AUTONOMIA] Intervenções detectadas: {contador_intervencoes} -> Nota: {n_autonomia}")
print(f" [QUALITY] Visão Objetos: {p_yolo:.1f}% | Tempo: {frame_idx} frames  | Média das outras métricas: {media_global_outros}-> Nota: {n_quality}")
print("-" * 60)
