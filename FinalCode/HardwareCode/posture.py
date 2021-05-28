#define led 3   
#define tiltSensor 8
#define buzzer 11     

int s=400;         
int v =0; 


void setup(){
   pinMode (led,OUTPUT);
   pinMode (tiltSensor,INPUT);
   pinMode(buzzer,OUTPUT);
}
void loop(){
  
  
   v = digitalRead(tiltSensor);
   if (v == HIGH){
     
      digitalWrite(led,HIGH);
      tone(buzzer,s);  
     
   }
   else {
     
   digitalWrite(led,LOW);
   noTone(buzzer);
     
   }
   delay(400);
}
