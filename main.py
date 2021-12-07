# -*- coding: utf-8 -*-
"""
@author: Augusto Manaro, Luis Larco, Thiago Torres
"""
#######################################################################################
## Imports 

import numpy as np
from sklearn.preprocessing import LabelEncoder
import tflite_runtime.interpreter as tflite
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
from scipy import signal  

import sounddevice as sd

import RPi.GPIO as IO
import os
import time       
          
###############################################################################
#                        SETUP

inicio_setup_time = time.time()
## Pin Configurations
pwm_pin_1 = 12
pwm_pin_2 = 13
botao = 1
led_verde = 7
led_azul = 8
led_vermelho = 25

## GPIO
IO.cleanup()
IO.setwarnings(False)           
IO.setmode (IO.BCM)

IO.setup(botao,IO.IN)           
IO.setup(led_verde,IO.OUT)
IO.setup(led_azul,IO.OUT)           
IO.setup(led_vermelho,IO.OUT)
IO.setup(pwm_pin_1,IO.OUT)
IO.setup(pwm_pin_2,IO.OUT)

pwm_1 = IO.PWM(pwm_pin_1,100)           
pwm_2 = IO.PWM(pwm_pin_2,100)  
pwm_1.start(0)         
pwm_2.start(0)  

print("Configuracoes de pinos realizadas com sucesso...")

## Neural Network Configurations
inicio = time.time()  
interpreter = tflite.Interpreter(model_path="models/model_v3.tflite")
interpreter.allocate_tensors()
importa_rede = time.time()

print('Rede Neural importada com sucesso...' )
le = LabelEncoder()
le.classes_ = np.load('models/classes_v1.npy')  
max_pad_len = 174
num_rows = 99
num_columns = 174
num_channels = 1
print('Classes importadas com sucesso...' )
fim_setup_time = time.time()
print("Configuracoes realizadas com sucesso! (%.2f s)" % (fim_setup_time - inicio_setup_time))

###############################################################################
#                       CHECKUP
#ChangeDutyCycle    
 
###############################################################################
#                        FUNCTIONS 

def extract_features(file_name):
    inicio_extract_time = time.time()
    try:
        audio = file_name.astype(np.float32, order='C') / 32768.00
        try:
            d = (audio[:,0] + audio[:,1]) / 2
            f = signal.resample(d, 22050)
        except:
            f = signal.resample(audio, 22050)
        mfccs = mfcc(f, samplerate =22050, numcep=40,nfft=1024)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        print(mfccs.shape)

    except Exception as e:
        print(e)
        return None
    fim_extract_time = time.time()
    print("Features extraidas -> %.2f s" % (fim_extract_time - inicio_extract_time)) 
    return mfccs


def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    inicio_prediction_time = time.time()
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    prediction_feature = np.float32(prediction_feature)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'],prediction_feature)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    fim_prediction_time = time.time()
    print("Categorizacao realizada -> %.2f s" % (fim_prediction_time - inicio_prediction_time)) 
    print(le.classes_[np.argmax(output_data)])


###############################################################################
##              MAIN

while True:                               
    try:
        IO.output(led_verde, IO.HIGH)
        ## Gravacao 4s
        
        fs = 44100  # Sample rate
        seconds = 4  # Duration of recording
        
        
        if IO.input(botao):
            IO.output(led_azul, IO.HIGH)            
            print("Gravando...") 
            inicio_gravacao_time = time.time()          
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()
            fim_gravacao_time = time.time()            
            print("fim gravacao")
            print("Tempo de gravacao -> %.2f s" % (fim_gravacao_time - inicio_gravacao_time))            
            IO.output(led_azul, IO.LOW)
                            
            print_prediction(myrecording)
            print("Aguardando..")
            
      
    
    except:        
        pwm_1.stop()
        pwm_2.stop()
        IO.cleanup()
        
IO.cleanup()        