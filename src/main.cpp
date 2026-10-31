#include <Arduino.h>
#include <time.h>
const unsigned int PIN_4 = 4;


void setup() {
  // put your setup code here, to run once:
  

  //Set the desired pin to be output.
  //OUTPUT is a macro defined by Arduino
  pinMode(PIN_4, OUTPUT);
  //Sets to communication rate between the board and serial.
  Serial.begin(9600);
  Serial.println("This program blinks a LED using PIN 4");
}

void loop() {
  // put your main code here, to run repeatedly:

  /*
    Switch between HIGH and LOW on the desired pin using digitalWrite function for a specified amount of time
    using delay function.

    HIGH and LOW are macros defined by arduino.
  */
  digitalWrite(PIN_4, HIGH);
  delay(500);
  digitalWrite(PIN_4, LOW);
  delay(500);
}