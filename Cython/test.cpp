#include<iostream>
#include<Python.h>
#include"Haar_cascade_detection.h"
using namespace std;

int main(){
 
  Py_Initialize();
  initHaar_cascade_detection();
  Run("haarcascade_frontalface_default.xml", "haarcascade_eye.xml", 1);
  Py_Finalize();
  return 0;
}
