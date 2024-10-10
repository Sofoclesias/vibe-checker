import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv.FONT_HERSHEY_SIMPLEX

'''
TO-DO
1. Configurar serial de Arduino para la conexión py-ino
https://www.instructables.com/Face-Tracking-Device-Python-Arduino/
'''

def camera_ready():
    '''
    Proyecta la cámara en una ventana de escritorio para la presentación expositiva.
    '''
    
    cap = cv.VideoCapture(0)
    cv.namedWindow("Vibe Checker",cv.WINDOW_NORMAL)
    center = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)//2),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)//2))
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            # Detección de rostros
            faces = face_cascade.detectMultiScale(cv.cvtColor(frame,cv.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                # Clasificación de la emoción en el rostro
                face = frame[y:y+h, x:x+w]
                
                '''
                TO-DO
                1. Aquí debería pasarse 'face' al modelo de clasificación para obtener la emoción               
                https://github.com/LanaDelSlay/Arduino-Face-Tracking/blob/main/face.py
                '''
                emotion = 'angry' # Cambiar por el resultado real del modelo
                acc = 0.69  # Cambiar por el resultado real del modelo

                # Dibujo de amenidades para la cámara
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame,f'{emotion} - {round(acc*100,2)}%',(x-5,y-15),font,0.8,(0, 255, 0),2)
                
                # Comandos para Arduino
                err = np.array(center) - np.array([[(y+h)//2],[(x+w)//2]])
                
                '''
                TO-DO
                1. Configurar el controlador PID para conseguir los ángulos de rotación para los pan-tilt
                2. Convertir lo anterior en señales analógicas para que Arduino las procese.
                3. Proyectar emoción en pantalla LDR
                
                https://github.com/Thomas9363/Face-Tracking-using-Mediapipe-Face-and-haarcascade-frontalface-with-or-without-PID-Control/blob/main/PanTiltMediapipeFacePID.py
                '''
            
                # Exportación de datos emocionales en .csv 
                '''
                TO-DO
                Hacer lo indicado xdddd
                
                En principio no debería registrar por cada frame de actividad, sino por cada cambio emocional o persona nueva detectada. De otra manera, se sobrecargaría el csv.
                Cranear ello más tarde zzz.
                '''
            
            cv.imshow('Vibe Checker',frame)
            
            if cv.waitKey(1)==ord('x'):
                break
        
    cap.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    camera_ready()