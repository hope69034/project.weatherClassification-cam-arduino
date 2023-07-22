
#include <Servo.h>  //서보모터 라이브러리 include

 

Servo microServo;  //서보모터 객체 선언

const int servoPin = 7;  //서보모터 제어핀 할당









const int LED_PIN_1 = 10; // First LED pin



const int LED_PIN_2 = 13; // Second LED pin



const int BUZZER_PIN = 8; // Active buzzer pin







void setup() {



 pinMode(LED_PIN_1, OUTPUT);



 pinMode(LED_PIN_2, OUTPUT);



 pinMode(BUZZER_PIN, OUTPUT);



 Serial.begin(9600);



   

  Serial.println("hello arduino!");

  microServo.attach(servoPin);  //서보모터 초기화

   



}







void loop() {



int angle;  //각도 변수 선언

//angle = 0; angle < 170



 if (Serial.available()) {



 String message = Serial.readStringUntil('\n');



 message.trim();







 if (message == "detected") {

  microServo.write(20);

  delay(1000); 

  microServo.write(130);

  // Blink the LEDs alternately



  for (int i = 0; i < 4; i++) {



  digitalWrite(LED_PIN_1, HIGH);



  digitalWrite(LED_PIN_2, LOW);



  digitalWrite(BUZZER_PIN, HIGH);



  delay(500);



  digitalWrite(LED_PIN_1, LOW);



  digitalWrite(LED_PIN_2, HIGH);



  digitalWrite(BUZZER_PIN, LOW);



  delay(500);



  }



  digitalWrite(LED_PIN_1, LOW);



  digitalWrite(LED_PIN_2, LOW);



  digitalWrite(BUZZER_PIN, LOW);



 }



 }



}