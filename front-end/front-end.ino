#include <Servo.h>

// TITULO:   SEGUIMIENTO CON OPEN CV Y ARDUINO

#define servomaxx 180      // ángulo máximo de rotación del servo pan 
#define servomaxy 180      // ángulo máximo de rotación del servo tilt 
#define screenmaxx 640     // resolución máxima en el eje x 
#define screenmaxy 480     // resolución máxima en el eje y 
#define servocenterx 90    // posición inicial del servo pan 
#define servocentery 0     // posición inicial del servo tilt 
#define servopinx 9        // pin digital para el servo pan
#define servopiny 10       // pin digital para el servo tilt
#define baudrate 57600     // velocidad de comunicación serial
#define distancex 1        // pasos de rotación en el eje x
#define distancey 1        // pasos de rotación en el eje y

int valx = 0;       // almacena el dato del eje x recibido del puerto serial
int valy = 0;       // almacena el dato del eje y recibido del puerto serial
int posx = 0;       // posición actual del servo en el eje x
int posy = 0;       // posición actual del servo en el eje y
int incx = 10;      // incremento significativo para el movimiento de la cámara en el eje x
int incy = 10;      // incremento significativo para el movimiento de la cámara en el eje y

Servo servox;       // objeto para el control del servo pan
Servo servoy;       // objeto para el control del servo tilt

short MSB = 0;      // byte más significativo (Most Significant Byte)
short LSB = 0;      // byte menos significativo (Least Significant Byte)
int MSBLSB = 0;     // combinación de MSB y LSB

void setup() {
  // Configuración inicial del sistema
  Serial.begin(baudrate);   // Inicializa la comunicación serial a la velocidad especificada
  Serial.println("Starting Cam-servo Face tracker");  // Mensaje de inicio

  pinMode(servopinx, OUTPUT);  // Configura el pin del servo pan como salida
  pinMode(servopiny, OUTPUT);  // Configura el pin del servo tilt como salida

  servoy.attach(servopiny);    // Asocia el objeto servo al pin del tilt
  servox.attach(servopinx);    // Asocia el objeto servo al pin del pan

  // Centra los servos en sus posiciones iniciales
  servox.write(servocenterx);  // Mueve el servo pan(90 grados)
  delay(200);
  servoy.write(servocentery);  // Mueve el servo tilt (0 grados)
  delay(200);
}

void loop() {
  while (Serial.available() <= 0); // Espera hasta que haya datos disponibles en el puerto serial
  if (Serial.available() >= 4)  // Espera hasta que haya al menos 4 bytes disponibles
  {
    // Obtener los datos de la coordenada X (2 bytes)
    MSB = Serial.read();         // Lee el byte más significativo
    delay(5);                    // Breve pausa para asegurar la lectura
    LSB = Serial.read();         // Lee el byte menos significativo
    MSBLSB = word(MSB, LSB);     // Combina los dos bytes en un solo valor
    valx = MSBLSB;               // Asigna el valor a la variable valx
    delay(5);

    // Obtener los datos de la coordenada Y (2 bytes)
    MSB = Serial.read();         // Lee el byte más significativo
    delay(5);
    LSB = Serial.read();         // Lee el byte menos significativo
    MSBLSB = word(MSB, LSB);     // Combina los dos bytes en un solo valor
    valy = MSBLSB;               // Asigna el valor a la variable valy
    delay(5);

    // Lee las posiciones actuales de los servos
    posx = servox.read(); 
    posy = servoy.read();

    // Determina si el componente X del rostro está a la izquierda del centro de la pantalla
    if (valx < (screenmaxx / 2 - incx)) {
      if (posx >= incx) posx += distancex; // Actualiza la posición del servo pan para moverse hacia la izquierda
    }
    // Determina si el componente X del rostro está a la derecha del centro de la pantalla
    else if (valx > screenmaxx / 2 + incx) {
      if (posx <= servomaxx - incx) posx -= distancex; // Actualiza la posición del servo pan para moverse hacia la derecha
    }

    // Determina si el componente Y del rostro está por debajo del centro de la pantalla
    if (valy < (screenmaxy / 2 - incy)) {
      if (posy >= 5) posy += distancey; // Si está por debajo, actualiza la posición del servo tilt para bajarlo
    }
    // Determina si el componente Y del rostro está por encima del centro de la pantalla
    else if (valy > (screenmaxy / 2 + incy)) {
      if (posy <= 175) posy -= distancey; // Si está por encima, actualiza la posición del servo tilt para subirlo
    }

    // Los servos se rotarán según las nuevas posiciones calculadas
    servox.write(posx);
    servoy.write(posy);
  }
}






