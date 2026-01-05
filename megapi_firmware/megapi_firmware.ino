#include <Arduino.h>
#include <MeMegaPi.h>

// ===== PORT MAPPING =====

// 1A->Motor1 (FR), 1B->Motor2 (BR), 2A->Motor3 (BL), 2B->Motor4 (FL)
MeMegaPiDCMotor motor1(PORT1A); // Front Right
MeMegaPiDCMotor motor2(PORT1B); // Back Right
MeMegaPiDCMotor motor3(PORT2A); // Back Left
MeMegaPiDCMotor motor4(PORT2B); // Front Left

// Sensors on the A6..A15 3-row header (SIG/5V/GND)
const uint8_t PIN_IR_L   = A6;  // IR Proximity Sensor 1
const uint8_t PIN_IR_C   = A7;  // IR Proximity Sensor 2
const uint8_t PIN_IR_R   = A8;  // IR Proximity Sensor 3

const uint8_t PIN_LINE_1 = A9;  // Line follower 1
const uint8_t PIN_LINE_2 = A10; // Line follower 2

const uint8_t PIN_IMPACT_1 = A11; // Impact switch 1
const uint8_t PIN_IMPACT_2 = A12; // Impact switch 2

// ===== HELPER FUNCTIONS =====

// MegaPi motor library expects motor speed in range [-255, +255]
static int clamp255(int v){ if(v>255) return 255; if(v<-255) return -255; return v; }

static void setAllMotors(int a,int b,int c,int d){
  motor1.run(clamp255(a));
  motor2.run(clamp255(b));
  motor3.run(clamp255(c));
  motor4.run(clamp255(d));
}

static void emergencyStop(){ setAllMotors(0,0,0,0); }

// Convert 0-1023 analog values to 0/1 with a threshold
static int asDigital(int analogValue){
  return (analogValue > 512) ? 1 : 0;
}

static void sendSensors(){
  int irL = analogRead(PIN_IR_L);
  int irC = analogRead(PIN_IR_C);
  int irR = analogRead(PIN_IR_R);

  int line1 = asDigital(analogRead(PIN_LINE_1));
  int line2 = asDigital(analogRead(PIN_LINE_2));

  int impact1 = asDigital(analogRead(PIN_IMPACT_1));
  int impact2 = asDigital(analogRead(PIN_IMPACT_2));

  Serial.print("S ");
  Serial.print(irL); Serial.print(' ');
  Serial.print(irC); Serial.print(' ');
  Serial.print(irR); Serial.print(' ');
  Serial.print(impact1); Serial.print(' ');
  Serial.print(impact2); Serial.print(' ');
  Serial.print(line1); Serial.print(' ');
  Serial.print(line2);
  Serial.print('\n');
}

// serial protocol parser (newline-terminated commands)
static bool readLine(char *buf, size_t maxLen){
  static size_t idx=0;
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\r') continue;
    // accumulates characters from Serial until a newline is encountered
    if(c=='\n'){ buf[idx]='\0'; idx=0; return true; }
    if(idx<maxLen-1) buf[idx++]=c;
  }
  return false;
}

// ===== SETUP & LOOP =====

void setup(){
  Serial.begin(115200);
  emergencyStop();
  delay(200);
  Serial.println("READY");
}

void loop(){
  char line[64];
  if(!readLine(line,sizeof(line))) return;

  if(line[0]=='M'){
    int a,b,c,d;
    if(sscanf(line,"M %d %d %d %d",&a,&b,&c,&d)==4){
      setAllMotors(a,b,c,d);
      Serial.println("OK");
    } else Serial.println("ERR");
  } else if(line[0]=='S'){
    sendSensors();
  } else if(line[0]=='E'){
    emergencyStop();
    Serial.println("OK");
  } else {
    Serial.println("ERR");
  }
}
