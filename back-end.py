import cv2
import serial
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.architecture import ResEmoteNet
from PIL import Image
import numpy as np
from datetime import datetime

# Configuración del clasificador Haar Cascade para la detección de rostros
face_cascade_name = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_name)

# Nombre de la ventana de visualización
window_name = "Vibe Checker"
font = cv2.FONT_HERSHEY_SIMPLEX

# Inicialización de PyTorch y el modelo predictivo para detección de emociones
dev = torch.device('cpu')  # Utiliza CPU como dispositivo
model = ResEmoteNet().to(dev)  # Carga el modelo en el dispositivo
checkpoint = torch.load('model/best.pth', map_location=dev, weights_only=True)  # Carga el modelo entrenado
model.load_state_dict(checkpoint['model_state_dict'])  # Carga los parámetros del modelo
model.eval()  # Configura el modelo en modo evaluación
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']  # Etiquetas de emociones

# Pipeline de preprocesamiento para las imágenes de entrada
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensiona la imagen a 64x64 píxeles
    transforms.Grayscale(num_output_channels=3),  # Convierte la imagen a escala de grises
    transforms.ToTensor(),  # Convierte la imagen en un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza los valores de los píxeles
])

# Archivo de registro para emociones detectadas
log_file = "emotion_log.txt"

def log_emotion(timestamp, emotion, accuracy):
    """
    Registra la emoción detectada junto con la fecha y hora en un archivo.
    """
    with open(log_file, "a") as log:
        log.write(f"[{timestamp}] {emotion} - {accuracy:.2f}%\n")
        print(f"[{timestamp}] {emotion} - {accuracy:.2f}%")  # Muestra el registro en la consola

def detect_emotion(frame):
    """
    Clasifica la emoción en un frame de rostro usando el modelo predictivo.
    """
    x = preprocess(frame).unsqueeze(0).to(dev)  # Preprocesa la imagen
    with torch.no_grad():  # Desactiva el cálculo de gradientes para mejorar el rendimiento
        y = model(x)  # Realiza la predicción
        probs = [round(score, 2) for score in F.softmax(y, dim=1).cpu().numpy().flatten()]  # Convierte las probabilidades a una lista
    idx = np.argmax(probs)  # Encuentra el índice con la probabilidad más alta
    return emotions[idx], probs[idx]  # Devuelve la emoción detectada y su probabilidad

# Configuración de la conexión serial con Arduino
arduino_com = serial.Serial(
    port='COM3', baudrate=57600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)

# Pausa para establecer la conexión serial
time.sleep(2)
arduino_com.close()  # Cierra la conexión para reiniciarla
time.sleep(2)
arduino_com.open()  # Abre la conexión nuevamente

def send_to_arduino(x, y):
    """
    Envía las coordenadas (x, y) al Arduino usando la conexión serial.
    """
    if arduino_com:
        try:
            # Envía las coordenadas X en dos bytes (LSB y MSB)
            LSB = x & 0xFF
            MSB = (x >> 8) & 0xFF
            arduino_com.write(bytes([MSB]))
            arduino_com.write(bytes([LSB]))
            
            # Envía las coordenadas Y en dos bytes (LSB y MSB)
            LSB = y & 0xFF
            MSB = (y >> 8) & 0xFF
            arduino_com.write(bytes([MSB]))
            arduino_com.write(bytes([LSB]))
        except Exception as e:
            print(f"Error enviando datos al Arduino: {e}")

def detect_and_display(frame):
    """
    Detecta rostros en el frame, clasifica la emoción y la muestra en pantalla.
    """
    # Convierte la imagen a escala de grises para mejorar la detección
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)  # Mejora el contraste de la imagen

    # Detecta rostros en la imagen
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)  # Calcula el centro del rostro detectado
        face = Image.fromarray(frame[y:y+h, x:x+w])  # Extrae el rostro como imagen
        emotion, prob = detect_emotion(face)  # Detecta la emoción del rostro
        global last_log_time 

        # Registra la emoción si ha pasado al menos 1 segundo desde el último registro
        current = time.time()
        if current - last_log_time >= 1:
            last_log_time = current
            log_emotion(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, prob * 100)
        
        # Dibuja un rectángulo alrededor del rostro y muestra la emoción
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotion} - {round(prob * 100, 2)}%', (x - 5, y - 15), font, 0.8, (0, 255, 0), 2)

        # Envía las coordenadas al Arduino
        send_to_arduino(center[0], center[1])

    # Muestra el resultado en la ventana
    cv2.imshow(window_name, frame)

# Variable global para el tiempo de registro de la emoción
last_log_time = 0

def main():
    # Captura de video desde la cámara (usar índice diferente si es necesario)
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("--(!)Error al abrir la cámara")
        return

    while True:
        ret, frame = capture.read()
        if frame is None:
            print("--(!) No se capturó el frame")
            break

        # Procesa y muestra el frame
        detect_and_display(frame)

        # Salir si se presiona la tecla 'c'
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

    # Libera los recursos de la cámara
    capture.release()
    cv2.destroyAllWindows()

    # Cierra la conexión serial
    if arduino_com:
        arduino_com.close()

if __name__ == "__main__":
    main()
