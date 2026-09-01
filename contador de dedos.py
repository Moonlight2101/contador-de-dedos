import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# --- ETAPA 1: CONFIGURAÇÃO INICIAL ---
# Define o caminho do modelo garantindo compatibilidade com qualquer SO
model_path = os.path.join(os.getcwd(), 'hand_landmarker.task')

# Configura as opções do detector de mãos do MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # Detectar apenas 1 mão
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# --- ETAPA 2: CAPTURA DE VÍDEO ---
# Abre a webcam (tente alterar para 1 caso o índice 0 não abra)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # Inverte o frame horizontalmente para efeito de espelho (mais intuitivo)
    frame = cv2.flip(frame, 1)
    
    # O OpenCV trabalha em BGR, mas o MediaPipe exige formato RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- ETAPA 3 & 4: DETECÇÃO DA MÃO E DESENHO ---
    # Converte o frame para o formato de imagem do MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Realiza a detecção dos landmarks
    detection_result = detector.detect(mp_image)
    
    dedos_levantados = 0

    # Verifica se alguma mão foi encontrada
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # Desenha os 21 pontos (verdes) e suas conexões na tela
            # Nota: O formato HandLandmarker do Tasks exige conversão para o formato clássico se usar o mp_draw diretamente.
            # Para cumprir os requisitos de desenho de forma nativa e simples no loop:
            h, w, _ = frame.shape
            
            # Lista para guardar as coordenadas reais convertidas em pixels
            pts = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pts.append((cx, cy))
                # Desenha o círculo verde em cada landmark
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            # Desenha as linhas de conexão (simplificado baseado nos índices do documento)
            conexoes = [
                 (0,1), (1,2), (2,3), (3,4),          # Polegar
                 (0,5), (5,6), (6,7), (7,8),          # Indicador
                 (5,9), (9,10), (10,11), (11,12),     # Médio
                 (9,13), (13,14), (14,15), (15,16),   # Anelar
                 (13,17), (17,18), (18,19), (19,20),  # Mindinho
                 (0,17) # Conexão da base da palma
            ]
            for start, end in conexoes:
                 cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)

            # --- ETAPA 5: LÓGICA DE CONTAGEM DOS DEDOS ---
            # Dedos normais (Indicador, Médio, Anelar, Mindinho)
            # Menor valor de Y significa que a ponta está mais alta na tela
            if pts[8][1] < pts[6][1]:   # Indicador
                dedos_levantados += 1
            if pts[12][1] < pts[10][1]: # Médio
                dedos_levantados += 1
            if pts[16][1] < pts[14][1]: # Anelar
                dedos_levantados += 1
            if pts[20][1] < pts[18][1]: # Mindinho
                dedos_levantados += 1

            # Lógica do Polegar (Tratamento de lateralidade)
            # Identifica se a mão é esquerda (Left) ou direita (Right)
            mao_tipo = detection_result.handedness[0][0].category_name
            
            if mao_tipo == "Right":
                if pts[4][0] > pts[2][0]: # Polegar direito esticado para fora
                    dedos_levantados += 1
            else:
                if pts[4][0] < pts[2][0]: # Polegar esquerdo esticado para fora
                    dedos_levantados += 1

    # --- ETAPA 6: EXIBIÇÃO DO RESULTADO ---
    # Cria o retângulo azul no canto superior esquerdo
    cv2.rectangle(frame, (20, 20), (120, 120), (255, 0, 0), cv2.FILLED)
    # Escreve a quantidade de dedos em branco dentro do retângulo
    cv2.putText(frame, str(dedos_levantados), (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    # Exibe o frame final na janela chamada "Imagem"
    cv2.imshow('Imagem', frame)

    # --- ETAPA 7: ENCERRAMENTO ---
    # Encerra o programa se o usuário pressionar a tecla ESC (código ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera os recursos do sistema
cap.release()
cv2.destroyAllWindows()