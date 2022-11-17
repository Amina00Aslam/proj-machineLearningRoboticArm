#include<Servo.h>
bool en_m1=true;
bool en_m2=true;
bool en_m3=true;

Servo myservo1;
Servo myservo2;
Servo myservo3;

void setup()
{
  Serial.begin(9600);
  myservo1.attach(3);
  myservo1.attach(5);
  myservo1.attach(6);
  delay(1000);

  myservo1.write(90); //claw
  myservo2.write(90); //rotation
  myservo3.write(90); //horizontal
  
}

void loop()
{
  if (Serial.available()>0)
  {
    String data_get=Serial.readString();
    String data_get_f=data_decode(data_get);
    Serial.println(data_get_f);

    if(data_get_f.toInt()<=180)
    {
      myservo3.write(180-data_get_f.toInt());
      Serial.println("cond_1");
    }
    else if(data_get_f=="2")
    {
      Serial.println("cond_2");
    }
    else if(data_get_f=="3")
    {
      Serial.println("cond_3");
    }
    else if(data_get_f=="500")
    {
      Serial.println("cond_3");
      myservo1.write(5);
      delay(2000);
      myservo1.write(90);
      delay(500);

      int drop_p=random(0,180);
      myservo3.write(180);
      delay(1000);
      myservo1.write(5);
      delay(1000);
      myservo1.write(90);
      delay(500);
      myservo3.write(90);
    }
  }
  delay(100);
}

String data_decode(String in_data)
{
  bool dump_en=false;
  String data_dump="";
  for(int i=0;i<in_data.length();i++)
  {
    if(in_data[i]=='#'&&dump_en==false)
    {
      dump_en=true;
    }
    else if(in_data[i]!='#'&&dump_en==true)
    {
      data_dump+=in_data[i];
    }
    else if(in_data[i]=='#'&&dump_en==true)
    {
      dump_en=false;
    }
  }
  return data_dump;
}
