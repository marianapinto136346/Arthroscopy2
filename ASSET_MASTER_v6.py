
# -*- coding: utf-8 -*-
"""
ASSET PROCESSOR 

"""

import cv2
import numpy as np
import os
import pandas as pd
import math
import sys
from ultralytics import YOLO
from collections import deque

# --- 1. CONFIGURAÇÕES ---
MODELO_YOLO_PATH = r'C:\Users\utilizador\Desktop\Pratica_Tese\Codigos\runs\segment\Tese_Mariana\Treino_Final_Graduacao\weights\best.pt'
PASTA_DEPTH = r'C:\Users\utilizador\Desktop\Pratica_Tese\Videos_Ensaio\profundidade'
PASTA_MOVIMENTO = r'C:\Users\utilizador\Desktop\Pratica_Tese\Videos_Ensaio\Dados_Movimento'
PASTA_MAE_SAIDA = r'C:\Users\utilizador\Desktop\Pratica_Tese\Videos_Ensaio\Videos_Ensaio'

AREA_MAXIMA = 45000
COOLDOWN_EVENTO = 3.0
LIMIAR_TOQUE = 5.0
LIMIAR_DANOS = 10.0
LIMIAR_BRUSCO_RAFT = 8.0
SEGUNDOS_ESPERA_INICIAL = 5.0
SEGUNDOS_MEMORIA = 3.0
LIMIAR_CONTATO_DEPTH = 12 
LIMIAR_DIF_ESTATICO = 1.2   # Sensibilidade de pixels (se a média de mudança for menor que isto, é o "mesmo frame")
TEMPO_LIMITE_ESTATICO = 7.0 # Segundos para imagem parada
TEMPO_LIMITE_AUSENCIA = 10.0 # Segundos sem ver NADA (ferramenta ou alvos)


# --- 2. SETUP DE ENTRADA E ARGUMENTOS DA UI ---
if len(sys.argv) > 2:
    VIDEO_PATH = sys.argv[1]
    # Captura as métricas enviadas pela UI (os argumentos após o path do vídeo)
    METRICAS_SELECIONADAS = sys.argv[2:]
else:
    # Fallback para execução manual sem UI
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    VIDEO_PATH = filedialog.askopenfilename(title="Selecionar Vídeo para Avaliação ASSET")
    if not VIDEO_PATH: sys.exit()
    # Se rodar manual, assume todas as métricas
    METRICAS_SELECIONADAS = ["Safety", "Field of View", "Camera Dexterity", "Instrument Dexterity", 
                             "Bi-Manual Dexterity", "Flow of Procedure", "Quality of Procedure", "Autonomy"]

# Se "Quality of Procedure" for selecionada, precisamos de todas as outras para o cálculo
precisa_todas = "Quality of Procedure" in METRICAS_SELECIONADAS


nome_base = os.path.basename(VIDEO_PATH).split('.')[0]
NOME_ALUNO = nome_base.replace('_RAFT_ANALISE', '')
path_csv = os.path.join(PASTA_MOVIMENTO, f"{NOME_ALUNO}_dados_flow.csv")
path_depth = os.path.join(PASTA_DEPTH, f"{NOME_ALUNO}_video_depth.mp4")

yolo_model = YOLO(MODELO_YOLO_PATH)
df_flow = pd.read_csv(path_csv) if os.path.exists(path_csv) else None
lista_bruscos_bool = (df_flow['magnitude_raft'] > LIMIAR_BRUSCO_RAFT).tolist() if df_flow is not None else []

cap = cv2.VideoCapture(VIDEO_PATH)
cap_depth = cv2.VideoCapture(path_depth) if os.path.exists(path_depth) else None
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# --- 3. CONFIGURAÇÃO DE RESOLUÇÕES DINÂMICAS ---
W_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

escala_x = W_ORIG / W_ORIG
escala_y = H_ORIG / H_ORIG

RAIO_FOV_CENTRO = int(H_ORIG * 0.15) # Cria um raio proporcional à resolução do vídeo

# print(f"Processamento: {W_ORIG}x{H_ORIG}")

# --- 4. CALIBRAÇÃO DINÂMICA (CORRIGIDA) ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000) 
ret, frame_ref = cap.read()
centro_calibrado = (W_ORIG / 2, H_ORIG / 2) 

if ret:
    frame_ref_res = cv2.resize(frame_ref, (W_ORIG, H_ORIG))
    gray_ref = cv2.cvtColor(frame_ref_res, cv2.COLOR_BGR2GRAY)
    blurred_ref = cv2.medianBlur(gray_ref, 7)
    
    # Ajustar dinamicamente os raios com base no tamanho do frame
    min_r = int(H_ORIG * 0.35)  # Ex: se H=1080, minRadius ~378
    max_r = int(H_ORIG * 0.55)  # Ex: se H=1080, maxRadius ~594
    
    circles = cv2.HoughCircles(blurred_ref, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150, 
                               param1=50, param2=35, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circles = np.float32(np.around(circles))
        centro_calibrado = (circles[0, 0][0], circles[0, 0][1])
  #      print(f"-> Centro calibrado com sucesso em: {centro_calibrado}")
    else:
        print("-> Círculo não detectado. A usar o centro geométrico padrão.")

# RESET REAL DOS VÍDEOS (Crucial para não dar erro de 0 frames)
cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)
if cap_depth:
    cap_depth.release()
    cap_depth = cv2.VideoCapture(path_depth)

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
    return 5 if v <= 4.2 else 4 if v <= 5.0 else 3 if v <= 5.5 else 2 if v <= 6.5 else 1

def atribuir_nota_raft(v):
    return 5 if v <= 0.95 else 4 if v <= 1.5 else 3 if v <= 2.0 else 2 if v <= 2.5 else 1

def desenhar_painel_hud(img, lista_textos, cores):
    """
    Desenha um painel preto translúcido e textos pequenos no canto superior esquerdo.
    """
    # Configurações de estilo
    FONTE = cv2.FONT_HERSHEY_SIMPLEX
    ESC = 0.55  # Tamanho da fonte (reduzido)
    ESP = 1     # Espessura da linha
    ALT_LINHA = 25
    MARGEM_X = 20
    MARGEM_Y = 30

    # 1. Desenhar fundo (opcional, mas ajuda na leitura)
    largura_painel = 300
    altura_painel = (len(lista_textos) * ALT_LINHA) + 20
    cv2.rectangle(img, (10, 10), (largura_painel, altura_painel), (0, 0, 0), -1)

    # 2. Escrever textos
    for i, (texto, cor) in enumerate(zip(lista_textos, cores)):
        pos_y = MARGEM_Y + (i * ALT_LINHA)
        cv2.putText(img, texto, (MARGEM_X, pos_y), FONTE, ESC, cor, ESP, cv2.LINE_AA)
        
        
cont_danos, cont_toque, cont_falhas_fluxo, cont_bruscos_fluxo = 0, 0, 0, 0
ultimo_evento_reg, ultima_falha_reg, frame_ultimo_brusco = -int(fps*COOLDOWN_EVENTO), -int(fps), -999
frames_com_yolo, frames_com_objeto_quality, frames_vistos_probe, frames_em_contato_alvo = 0, 0, 0, 0
historico_jitter_instr, historico_jitter_camera, historico_fov_dist, historico_centro_fov = [], [], [], []
historico_luz_fov, historico_ang_cam = [], []
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
ja_moveu_apos_paragem = False
buffer_mov_autonomia = deque(maxlen=15) # Analisa meio segundo de vídeo para decidir se parou
frames_ausencia_consecutivos = 0
frames_estaticos_consecutivos = 0


ret, f_init = cap.read()
prev_gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f_init, (W_ORIG, H_ORIG)), cv2.COLOR_BGR2GRAY), (7,7), 0)

# --- 7. LOOP DE PROCESSAMENTO ---
frame_idx = 0
MARGIN_WIDTH = 50

while cap.isOpened():
    ret, frame_raw = cap.read()
    if not ret: break

    # ==========================================
    # --- ZONA DE INICIALIZAÇÃO ANTI-CRASH ---
    # ==========================================
    brilho = 0.0
    dist_c = 0.0
    erro_horizonte = 0.0
    jitter_inst_atual = 0.0
    validar_contato = False
    status_flow = "NAVEGACAO"
    pt_orig = (0, 0)
    instabilidade_txt = "Instabilidade camara: -"
    horizontalidade_txt = "Horizontalidade: -"
    dist_centro_camara_txt = "Distancia ao centro: -"
    nota_luz_txt = "Luz: -"
    dist_txt = "Distancia ao centro: -"
    
    frame_proc = cv2.resize(frame_raw, (W_ORIG, H_ORIG))
    frame_draw = frame_raw.copy() # Canvas em Alta Resolução
    
    # 1. Depth Mapping
    ret_d, frame_d = cap_depth.read() if cap_depth else (False, None)
    if ret_d and frame_d is not None:
        largura_original_d = (frame_d.shape[1] - MARGIN_WIDTH) // 2
        mapa_depth = frame_d[:, largura_original_d + MARGIN_WIDTH:]
        depth_gray = cv2.cvtColor(cv2.resize(mapa_depth, (W_ORIG, H_ORIG)), cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = None

    # 2. RAFT / Bruscos
    if frame_idx < len(lista_bruscos_bool) and lista_bruscos_bool[frame_idx]:
        if frame_idx - frame_ultimo_brusco > 60:
            cont_bruscos_fluxo += 1
            frame_ultimo_brusco = frame_idx

    # 3. YOLO Segmentação
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
                alvo_ja_foi_visto = True
                frames_com_objeto_quality += 1
                cv2.fillPoly(mask_alvos, [pts], 255)
                x1, y1, x2, y2 = box.xyxy[0]
                alvos_atuais[label] = ((x1+x2)/2, (y1+y2)/2)
                roi = cv2.cvtColor(frame_proc[max(0,int(y1)):min(H_ORIG,int(y2)), max(0,int(x1)):min(W_ORIG,int(x2))], cv2.COLOR_BGR2GRAY)
                if roi.size > 0:
                    brilho = float(np.mean(roi))
                    historico_brilho_bruto.append(brilho)

    historico_probe_recent.append(probe_no_frame)

    # 4. Safety (Optical Flow)
    gray_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_proc, (7,7), 0)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_blur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    historico_jitter_camera.append(np.mean(mag))

    mag[mag < 0.8] = 0
    mov_global = np.median(mag)
    
    # LÓGICA DE AUTONOMIA MATEMÁTICA
    nomes_alvos = ["green_screw", "red_triangle", "red_cross"]
    alvo_visivel = any(name in alvos_atuais for name in nomes_alvos)
    presenca_detetada = probe_no_frame or alvo_visivel
    
    diff_frames = cv2.absdiff(prev_gray, gray_blur)
    mudanca_visual = np.mean(diff_frames)
    imagem_parada = mudanca_visual < LIMIAR_DIF_ESTATICO

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
    
    # Criar máscara de análise (FOV central)
    mask_analise = np.zeros((H_ORIG, W_ORIG), dtype=np.uint8)
    cv2.circle(mask_analise, (int(centro_calibrado[0]), int(centro_calibrado[1])), int(H_ORIG//2.5), 255, -1)
    
    mask_alvos_dilatada = cv2.dilate(mask_alvos, np.ones((10,10), np.uint8))
    mask_tecido_limpa = cv2.bitwise_and(mask_analise, cv2.bitwise_not(mask_alvos_dilatada))
    mask_tecido_limpa = cv2.bitwise_and(mask_tecido_limpa, cv2.bitwise_not(mask_probe))
    
    mags_tecido = mag[mask_tecido_limpa > 0]
    mov_local = np.percentile(mags_tecido, 95) if mags_tecido.size > 0 else 0
    
    probe_recente_ativa = any(historico_probe_recent)
    contato_com_alvo = np.any(cv2.bitwise_and(mask_probe, mask_alvos))

    if (frame_idx / fps) >= SEGUNDOS_ESPERA_INICIAL and alvo_ja_foi_visto and probe_recente_ativa and not contato_com_alvo:
        if mov_local > (mov_global + LIMIAR_DANOS) or mov_local > (mov_global + LIMIAR_TOQUE):
            if frame_idx - ultimo_evento_reg > (fps * COOLDOWN_EVENTO):
                if mov_local > (mov_global + LIMIAR_DANOS):
                    cont_danos += 1
                else:
                    cont_toque += 1
                ultimo_evento_reg = frame_idx

    # ==========================================
    # --- CÁLCULOS MATEMÁTICOS UNIFICADOS ---
    # ==========================================

    # Cálculo do FLOW OF PROCEDURE
    if probe_no_frame and alvos_atuais:
        M_p = cv2.moments(mask_probe)
        M_a = cv2.moments(mask_alvos)
        if M_p["m00"] != 0 and M_a["m00"] != 0:
            cX_p, cY_p = int(M_p["m10"] / M_p["m00"]), int(M_p["m01"] / M_p["m00"])
            cX_a, cY_a = int(M_a["m10"] / M_a["m00"]), int(M_a["m01"] / M_a["m00"])
            distancia_centros = math.sqrt((cX_p - cX_a)**2 + (cY_p - cY_a)**2)

            if depth_gray is not None:
                std_p = np.std(depth_gray[mask_probe == 255])
                std_a = np.std(depth_gray[mask_alvos == 255])
                intersecao = cv2.bitwise_and(mask_probe, mask_alvos)

                if np.any(intersecao) or (abs(std_p - std_a) < 2.0 and distancia_centros <= 10.0):
                    validar_contato = True
                    status_flow = "CONTACTO"
                    frames_em_contato_alvo += 1
                else:
                    status_flow = "FALHA"
                    if frame_idx - ultima_falha_reg > 3 * fps:
                        cont_falhas_fluxo += 1
                        ultima_falha_reg = frame_idx

    # Cálculo do CAMERA DEXTERITY
    c_x_orig, c_y_orig = int(centro_calibrado[0]*escala_x), int(centro_calibrado[1]*escala_y)
    if alvos_atuais:
        label, pt = list(alvos_atuais.items())[0]
        pt_orig = (int(pt[0]*escala_x), int(pt[1]*escala_y))
        dist_c = math.sqrt((pt[0]-centro_calibrado[0])**2 + (pt[1]-centro_calibrado[1])**2)
        historico_fov_dist.append(dist_c)
        
        dx, dy = pt[0]-centro_calibrado[0], -(pt[1]-centro_calibrado[1])
        ang = float(np.degrees(np.arctan2(float(dy), float(dx)))) % 360
        ref = min([0, 90, 180, 270], key=lambda x: min(abs(x-ang), 360-abs(x-ang)))
        erro_horizonte = min(abs(ang-ref), 360-abs(ang-ref))
        historico_ang_cam.append(erro_horizonte)
        
    jitter_cam_atual = historico_jitter_camera[-1] if historico_jitter_camera else 0.0
    instabilidade_txt = f"Instabilidade camara: {jitter_cam_atual:.2f}"
    horizontalidade_txt = f"Horizontalidade: {erro_horizonte:.1f}"
    dist_centro_camara_txt = f"Distancia ao centro: {dist_c:.1f}px"

    # Cálculo do FIELD OF VIEW (Strings)
    nota_luz_txt = f"Luz: {brilho:.1f}"
    dist_txt = f"Distancia ao centro: {dist_c:.1f}px"

    # Cálculo do INSTRUMENT DEXTERITY
    if centro_probe_atual and pos_ant_probe:
        dist_p = math.sqrt((centro_probe_atual[0]-pos_ant_probe[0])**2 + (centro_probe_atual[1]-pos_ant_probe[1])**2)
        if 0.5 < dist_p < 80:
            jitter_inst_atual = dist_p
            historico_jitter_instr.append(jitter_inst_atual)
    perc_direcionamento = (frames_em_contato_alvo / frames_vistos_probe * 100) if frames_vistos_probe > 0 else 0.0

    # Máquina de Estados BI-MANUAL
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
            else:
                contador_ausencia_total = 0 
            if contador_ausencia_total > LIMIAR_AUSENCIA:
                estado_bimanual = 2
                fase_transicao = "FINALIZADO"

    jitter_atual = jitter_inst_atual if probe_no_frame else 0.0
    raft_atual = df_flow.iloc[frame_idx]['magnitude_raft'] if (df_flow is not None and frame_idx < len(df_flow)) else 0.0

    dados_bimanual[estado_bimanual]["instr"].append(jitter_inst_atual)
    dados_bimanual[estado_bimanual]["raft"].append(raft_atual)

    tempo_estatico = frames_estaticos_consecutivos / fps
    tempo_ausencia = frames_ausencia_consecutivos / fps
    cor_e = (0, 0, 255) if tempo_estatico > TEMPO_LIMITE_ESTATICO else (255, 255, 255)
    cor_a = (0, 0, 255) if tempo_ausencia > TEMPO_LIMITE_AUSENCIA else (255, 255, 255)

    # ==========================================
    # --- RENDERIZAÇÃO E GRAVAÇÃO DE VÍDEOS ---
    # ==========================================
    
    # HUD 1: SAFETY
    if video_safety_out is not None:
        frame_s = frame_draw.copy()
        textos_s = [f"DANOS: {cont_danos}", f"TOQUES: {cont_toque}"]
        cores_s = [(0, 0, 255), (0, 165, 255)]
        desenhar_painel_hud(frame_s, textos_s, cores_s)
        video_safety_out.write(frame_s)

    # HUD 2: FLOW OF PROCEDURE
    if video_flow_out is not None:
        frame_f = frame_draw.copy()
        status_cor = (0, 255, 0) if status_flow == "CONTACTO" else (0, 0, 255) if status_flow == "FALHA" else (255, 255, 255)
        cor_b = (0, 165, 255) if (frame_idx - frame_ultimo_brusco < 30) else (255, 255, 255)
        textos_f = [f"Estado: {status_flow}", f"Falhas relativas ao alvo: {cont_falhas_fluxo}", f"Mov. Bruscos: {cont_bruscos_fluxo}"]
        cores_f = [status_cor, (255, 255, 255), cor_b]
        desenhar_painel_hud(frame_f, textos_f, cores_f)
        video_flow_out.write(frame_f)

    # HUD 3: CAMERA
    if video_camera_out is not None:
        frame_c = frame_draw.copy()
        textos_c = [instabilidade_txt, horizontalidade_txt, dist_centro_camara_txt]
        cores_c = [(0, 255, 255), (255, 255, 0), (0, 255, 0)]
        desenhar_painel_hud(frame_c, textos_c, cores_c)
        if alvos_atuais:
            cv2.line(frame_c, (c_x_orig, c_y_orig), pt_orig, (255, 0, 0), 2)
        cv2.drawMarker(frame_c, (c_x_orig, c_y_orig), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame_c, (c_x_orig, c_y_orig), int(RAIO_FOV_CENTRO * escala_x), (255, 255, 255), 1)
        video_camera_out.write(frame_c)

    # HUD 4: FIELD OF VIEW
    if video_fov_out is not None:
        frame_fov = frame_draw.copy()
        if results[0].masks:
            for mask_data, box in zip(results[0].masks.xy, results[0].boxes):
                pts_f = mask_data.astype(np.int32)
                if cv2.contourArea(pts_f) > AREA_MAXIMA: continue
                label_f = yolo_model.names[int(box.cls)].lower()
                if any(t in label_f for t in ["green_screw", "red_triangle", "red_cross"]):
                    cv2.polylines(frame_fov, [(pts_f * [escala_x, escala_y]).astype(np.int32)], True, (0, 255, 0), 2)
        textos_fov = [nota_luz_txt, dist_txt]
        cores_fov = [(0, 255, 0), (0, 255, 255)]
        desenhar_painel_hud(frame_fov, textos_fov, cores_fov)
        cv2.drawMarker(frame_fov, (c_x_orig, c_y_orig), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame_fov, (c_x_orig, c_y_orig), int(RAIO_FOV_CENTRO * escala_x), (255, 255, 255), 1)
        video_fov_out.write(frame_fov)

    # HUD 5: INSTRUMENT
    if video_instr_out is not None:
        frame_i = frame_draw.copy()
        if results[0].masks:
            for mask_data, box in zip(results[0].masks.xy, results[0].boxes):
                pts_i = mask_data.astype(np.int32)
                if cv2.contourArea(pts_i) > AREA_MAXIMA: continue
                label_i = yolo_model.names[int(box.cls)].lower()
                pts_rescaled = (pts_i * [escala_x, escala_y]).astype(np.int32)
                if "probe" in label_i:
                    cv2.polylines(frame_i, [pts_rescaled], True, (0, 255, 0), 2)
                elif any(t in label_i for t in ["green_screw", "red_triangle", "red_cross"]):
                    cor_alvo = (0, 0, 255) if validar_contato else (255, 255, 255) 
                    cv2.polylines(frame_i, [pts_rescaled], True, cor_alvo, 2)
        status_instr = "CONTACTO" if validar_contato else "NAVEGACAO"
        cor_status = (0, 255, 0) if validar_contato else (0, 165, 255)
        textos_i = [f"Instabilidade Instrumento: {jitter_inst_atual:.2f}", f"Corretamente Orientada: {perc_direcionamento:.1f}%", f"Estado: {status_instr}"]
        cores_i = [(255, 255, 255), (255, 255, 255), cor_status]
        desenhar_painel_hud(frame_i, textos_i, cores_i)
        video_instr_out.write(frame_i)

    # HUD 6: BI-MANUAL
    if video_bimanual_out is not None:
        frame_b = frame_raw.copy()
        mao_nome = "DIREITA" if estado_bimanual == 1 else "ESQUERDA"
        cor_mao = (255, 128, 0) if estado_bimanual == 1 else (255, 0, 255)
        textos_b = [f"MAO: {mao_nome}", f"Instabilidade Instrumento: {jitter_atual:.2f}", f"Instabilidade Camara: {raft_atual:.2f}"]
        cores_b = [cor_mao, (200, 200, 200), (255, 255, 255)]
        desenhar_painel_hud(frame_b, textos_b, cores_b)
        if centro_probe_atual:
            letra = "D" if estado_bimanual == 1 else "E"
            cv2.putText(frame_b, letra, (int(centro_probe_atual[0]*escala_x) + 15, int(centro_probe_atual[1]*escala_y) - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_mao, 3)
        video_bimanual_out.write(frame_b)

    # HUD 7: AUTONOMY
    if video_autonomia_out is not None:
        frame_a = frame_draw.copy()
        textos_a = [
            f"INTERVENCOES: {contador_intervencoes}",
            f"Camara Estatica: {tempo_estatico:.1f}s / {TEMPO_LIMITE_ESTATICO}s",
            f"Ausencia de alvos: {tempo_ausencia:.1f}s / {TEMPO_LIMITE_AUSENCIA}s",
        ]
        cores_a = [(255, 255, 255), cor_e, cor_a]
        desenhar_painel_hud(frame_a, textos_a, cores_a)
        video_autonomia_out.write(frame_a)

    # Atualizações de Fim de Frame
    if centro_probe_atual:
        pos_ant_probe = centro_probe_atual
    prev_gray = gray_blur.copy()
    frame_idx += 1 

# --- FIM DO LOOP WHILE ---

    
# --- 6. CÁLCULO DAS NOTAS FINAIS ---
if frame_idx == 0: sys.exit("Erro: Vídeo sem frames processados.")

# [SAFETY]
if (cont_danos <=1 and cont_toque <= 3): n_safety = 5
elif cont_danos <= 4 and cont_toque <= 7: n_safety = 4
elif cont_danos <= 8 and cont_toque <= 20: n_safety = 3
elif cont_danos <= 10 and cont_toque <= 30: n_safety = 2
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

p_yolo = (frames_com_yolo / frame_idx) * 100
n_vis = 5 if p_yolo >= 80 else 4 if p_yolo >= 75 else 3 if p_yolo >= 55 else 2 if p_yolo >= 50 else 1
n_bim = int(round((n_mov_global + n_vis) / 2))

# [CAMERA DEXTERITY]
h_med = np.mean(historico_ang_cam) if historico_ang_cam else 99
f_med = np.mean(historico_fov_dist) if historico_fov_dist else 999
j_med_cam = np.mean(historico_jitter_camera) if historico_jitter_camera else 99

if p_yolo >= 75 and h_med <= 25 and f_med <= 138 and j_med_cam <= 3.05: n_camera = 5
elif p_yolo >= 70 and h_med <= 30 and f_med <= 140 and j_med_cam <= 3.0: n_camera = 4
elif p_yolo >= 65 and h_med <= 35 and f_med <= 150 and j_med_cam <= 5.0: n_camera = 3 
elif p_yolo >= 60 and h_med <= 40 and f_med <= 160 and j_med_cam <= 7.0: n_camera = 2
else: n_camera = 1

# [INSTRUMENT DEXTERITY]
hesitacao = np.mean(historico_jitter_instr) + (np.std(historico_jitter_instr) * 0.5) if historico_jitter_instr else 99
perc_contato = (frames_em_contato_alvo / frames_vistos_probe * 100) if frames_vistos_probe > 0 else 0
n_f = 5 if hesitacao <= 10.5 else 4 if hesitacao <= 16.0 else 3 if hesitacao < 18.0 else 2 if hesitacao < 20.0 else 1
n_d = 5 if perc_contato > 55 else 4 if perc_contato > 45 else 3 if perc_contato > 35 else 2 if perc_contato > 30 else 1
n_instr = int(round((n_f * 0.5) + (n_d * 0.5)))

# [FIELD OF VIEW]
m_c = np.mean(historico_fov_dist) if historico_fov_dist else 0
m_l = np.mean(historico_brilho_bruto) if historico_brilho_bruto else 0
# Na artroscopia, o brilho ideal situa-se entre 90 e 150. A distância ideal é dentro do raio central.
if p_yolo >= 74 and m_c <= RAIO_FOV_CENTRO and (90 <= m_l <= 150): 
    n_fov = 5
elif p_yolo >= 65 and m_c <= (RAIO_FOV_CENTRO * 1.5) and (75 <= m_l <= 170): 
    n_fov = 4
elif p_yolo >= 55 and m_c <= (RAIO_FOV_CENTRO * 2.0) and (60 <= m_l <= 185): 
    n_fov = 3
elif p_yolo >= 50 and m_c <= (RAIO_FOV_CENTRO * 2.5) and (40 <= m_l <= 210): 
    n_fov = 2
else: 
    n_fov = 1


# [FLOW OF PROCEDURE]
if cont_falhas_fluxo <= 16 and cont_bruscos_fluxo <= 4 and p_yolo >= 80 and frame_idx <= 6300: n_flow = 5
elif cont_falhas_fluxo <= 30 and cont_bruscos_fluxo <= 5 and p_yolo >= 70 and frame_idx <= 9000: n_flow = 4
elif cont_falhas_fluxo <= 40 and cont_bruscos_fluxo <= 5 and p_yolo >= 65 and frame_idx <= 12600: n_flow = 3
elif cont_falhas_fluxo <= 50 and cont_bruscos_fluxo <= 5 and p_yolo >= 55 and frame_idx <= 18000: n_flow = 3
else: n_flow = 1

# [QUALITY OF PROCEDURE]
p_visao_obj = (frames_com_objeto_quality / frame_idx) * 100
n_visao = 5 if p_visao_obj >= 75 else 4 if p_visao_obj >= 65 else 3 if p_visao_obj >= 50 else 2 if p_visao_obj >= 40 else 1
n_tempo = 5 if 1800 <= frame_idx <= 6300 else 4 if 6300 < frame_idx <= 9000 else 3 if 9000 < frame_idx <= 12600 else 2 if 12600 < frame_idx <= 18000 else 1
#n_quality = int(round((n_visao + n_tempo) / 2))


# [AUTONOMIA]
# 1 - Incapaz (Se o vídeo terminar parado por muito tempo ou excesso de intervenções - ajuste conforme critério)
# 2 - Capaz com intervenções
# 3 - Capaz sem intervenções
n_autonomia = 1

if contador_intervencoes <= 3:
    n_autonomia = 3
elif contador_intervencoes < 5:
    n_autonomia = 2
elif contador_intervencoes <= 8: 
    n_autonomia = 1
    
    
    
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

# --- 7. FINALIZAÇÃO ---
cap.release()
if cap_depth: 
    cap_depth.release()

# Fecha todos os writers uma única vez
if video_safety_out: video_safety_out.release()
if video_camera_out: video_camera_out.release()
if video_instr_out: video_instr_out.release()
if video_bimanual_out: video_bimanual_out.release()
if video_flow_out: video_flow_out.release()
if video_fov_out: video_fov_out.release()
if video_autonomia_out: video_autonomia_out.release()

cv2.destroyAllWindows()


# --- 8. DASHBOARD TERMINAL ---
print("\n" + "="*60)
print(f" RELATÓRIO TÉCNICO INTEGRADO ASSET: {NOME_ALUNO}")
print("="*60)
print(f" [SAFETY] Danos: {cont_danos} | Toques: {cont_toque} -> Nota: {n_safety}")
print(f" [FOV] Centro Médio: {m_c:.2f} | Luz: {m_l:.2f} -> Nota: {n_fov}")
print(f" [CAMARA] Erro horizonte: {h_med:.2f} | Jitter: {j_med_cam:.2f} -> Nota: {n_camera}")
print(f" [INSTRUMENT] Hesitação: {hesitacao:.2f} | Direcionamento/contacto: {perc_contato:.1f}% -> Nota: {n_instr}")
print(f" [BIMANUAL] Jitter D/E: {j_dir:.2f}/{j_esq:.2f} | Nota: {n_bim}")
print(f"            RAFT D/E:   {r_dir:.2f}/{r_esq:.2f} -> Nota: {n_bim}")
print(f" [FLOW] Falhas: {cont_falhas_fluxo} | Mov bruscos: {cont_bruscos_fluxo} -> Nota: {n_flow}")
print(f" [AUTONOMIA] Intervenções detectadas: {contador_intervencoes} -> Nota: {n_autonomia}")
print(f" [QUALITY] Visão Objetos: {p_visao_obj:.1f}% | Tempo: {frame_idx} frames  | Média das outras métricas: {media_global_outros}-> Nota: {n_quality}")
print("-" * 60)

