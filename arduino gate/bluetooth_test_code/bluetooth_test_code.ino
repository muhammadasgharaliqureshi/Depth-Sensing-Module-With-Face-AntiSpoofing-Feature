
#include <Servo.h>
int counter = 0;
Servo myservo;  
int pos = 0;
void setup() {
  myservo.attach(9);
Serial.begin(9600);




}
void loop() {
char received=0;     
 
  if(Serial.available()>=0){
    received=Serial.read();
    delay(100);
    Serial.println(received);
    }
        
      if(received=='O'){
        for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    //delay(10);// waits 15ms for the servo to reach the position
  }
        counter = 1;
        }        
      else if(received=='C'){
        if (counter == 1){
         for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    //delay(10);                       // waits 15ms for the servo to reach the position
  }
         counter = 0;
         }
        }
        }
