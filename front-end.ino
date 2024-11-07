#include <Servo.h>

Servo panServo;
Servo tiltServo;

int panPin = 9;
int tiltPin = 10;

void setup() {
    Serial.begin(9600);
    panServo.attach(panPin);
    tiltServo.attach(tiltPin);

    panServo.write(90);
    tiltServo.write(90);
}

void loop() {
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil(\n);
        int comma = data.indexOf(',');

        if (comma > 0) {
            int panAngle = data.substring(0, comma).toInt();
            int tiltAngle = data.substring(comma + 1).toInt();

            panServo.write(panAngle);
            tiltServo.write(tiltAngle);
        }
    }
}