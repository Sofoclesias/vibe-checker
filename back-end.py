import cv2 as cv
import numpy as np
import serial
import time
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.architecture import ResEmoteNet

# https://www.instructables.com/Face-Tracking-Device-Python-Arduino/

arduino = serial.Serial(
    port='COM3', baudrate=9600, bytesize=8, timeout=2,stopbits=serial.STOPBITS_ONE               
)
time.sleep(2)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv.FONT_HERSHEY_SIMPLEX

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

preprocess = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ResEmoteNet().to(dev)
model.load_state_dict(torch.load('model/best.pth', weights_only=True)['model_state_dict'])
model.eval()

# Hiperparámetros PID controller
Kp = 0.5 
Ki = 0.01
Kd = 0.1

def send_signal(pan, tilt):
    com = f"{pan},{tilt}\n"
    arduino.write(com.encode())

def pid_control(err,prev_err,integral):
    integral += err
    output = Kp * err + Ki * integral + Kd * (err - prev_err)
    return output.flatten(), integral

def detect_emotion(frame):
    x = preprocess(frame).unsqueeze(0).to(dev)
    with torch.no_grad():
        y = model(x)
        probs = [round(score,2) for score in F.softmax(y,dim=1).cpu().numpy().flatten()]
    idx = np.argmax(probs)
    return emotions[idx], probs[idx]

def camera_ready():
    '''
    Proyecta la cámara en una ventana de escritorio para la presentación expositiva.
    '''
    
    cap = cv.VideoCapture(0)
    cv.namedWindow("Vibe Checker",cv.WINDOW_NORMAL)
    center = np.array([[int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//2)],[int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//2)]])
    prev_err = np.array([[0],[0]])
    integral = np.array([[0],[0]])

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame,1)

        if ret:
            # Detección de rostros
            faces = face_cascade.detectMultiScale(cv.cvtColor(frame,cv.COLOR_BGR2GRAY), scaleFactor=1.05, minNeighbors=20, minSize=(48, 48), flags=cv.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                # Clasificación de la emoción en el rostro
                face = Image.fromarray(frame[y:y+h, x:x+w])
                emotion, prob = detect_emotion(face)

                # Dibujo de amenidades para la cámara
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame,f'{emotion} - {round(prob*100,2)}%',(x-5,y-15),font,0.8,(0, 255, 0),2)
                
                # Comandos para Arduino
                err = np.array(center) - np.array([[(x+w)//2],[(y+h)//2]])
                rotation, integral = pid_control(err,prev_err,integral)
                prev_err = err

                send_signal(rotation[0],rotation[1])
            
            cv.imshow('Vibe Checker',frame)
            if cv.waitKey(1)==ord('x'):
                break
        
    cap.release()
    cv.destroyAllWindows()
    arduino.close()

if __name__=='__main__':
    camera_ready()