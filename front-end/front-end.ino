#include <LiquidCrystal_I2C.h>
#include <Servo.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);  // Adjust 0x27 to your I2C address if necessary

Servo panServo;
Servo tiltServo;

const int panPin = 3;   // Servo connected to PWM pin ~3
const int tiltPin = 5;  // Servo connected to PWM pin ~5

void setup() {
    Serial.begin(9600);
    panServo.attach(panPin);
    tiltServo.attach(tiltPin);

    panServo.write(90);  // Set initial position to 90 degrees
    tiltServo.write(90); // Set initial position to 90 degrees

    lcd.init();          // Initialize the LCD
    lcd.backlight();     // Turn on the backlight if available
    lcd.clear();         // Clear the LCD screen
}

void loop() {
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil('\n');
        lcd.clear();             // Clear previous text
        
        if (data.length() > 0) {
            int firstComma = data.indexOf(',');
            int secondComma = data.indexOf(',', firstComma + 1);

            if (firstComma > 0 && secondComma > firstComma) {
                String emotion = data.substring(0, firstComma);
                int panAngle = data.substring(firstComma + 1, secondComma).toInt();
                int tiltAngle = data.substring(secondComma + 1).toInt();

                lcd.print(emotion);

                panServo.write(panAngle);
                tiltServo.write(tiltAngle);
            }
        }
        
    }
}

