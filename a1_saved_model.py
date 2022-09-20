


import sys
import os
import numpy as np
from tensorflow import keras

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
learning_rate=0.01
input_layer = Input(shape=(2,))
dense_layer_1 = Dense(10, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(10, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)
model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
#model = keras.models.load_model(r'C:\Users\Hamed\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)')
memory=[]




from socket import AF_INET,SOCK_STREAM,socket
workingdirectory=os.getcwd()




def addition(num1,num2):
    
    sumof=float(num1)+0.01*int(num2)
    return sumof


socket = socket(AF_INET,SOCK_STREAM)
socket.bind(('127.0.0.1',int(sys.argv[1])))
socket.listen(1)
sock, addr = socket.accept()

while True:
    data = sock.recv(16384)
    text = data.decode('utf-8') 
    nums=text.split(":")
    
    inp=[float(nums[0]),int(nums[1])]
    oup=[float(addition(float(nums[0]),int(nums[1])))]
    memory.extend([[inp,oup]])
    inputs = np.array([i[0] for i in memory])
    outputs= np.array([i[1] for i in memory])
    model.fit(inputs,outputs, epochs=1, verbose=0)
    ans= model.predict(np.array([[float(nums[0]),int(nums[1])]]))[0][0]
    
    ans=str(ans)+"\n"
    sock.send(ans.encode('utf-8'))