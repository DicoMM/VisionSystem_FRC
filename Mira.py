import os
import cv2
import numpy as np
import time
from networktables import NetworkTables
import threading
import time



 #Define o nome dos arquivos que dever ser utilizados em conjunto com a programação
MODEL_NAME = ('modelo1')
GRAPH_NAME = ('detect.tflite')
LABELMAP_NAME = ('label_map.txt')

# Define a % minima de identificação do modelo
min_conf_threshold = float(0.7)
# Define a resolução da transmissão
imW, imH = int(854), int(480)

# Importa as bibliotecas do Tensorflow Lite
from tflite_runtime.interpreter import Interpreter

# Caminho da pasta atual
CWD_PATH = os.getcwd()

# Caminho para o arquivo .tflite, ele contem o modelo utilizado pra o reconhecimento de imagem
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# caminho para o arquivo label map
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Carrega o Label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


if labels[0] == '???':
    del(labels[0])

# Carrega o modelo treinado
if True:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Extrai os detalhes do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize video stream
videostream = cv2.VideoCapture(0)
time.sleep(1)

#Chama a tabela "smart dashboard"
table = NetworkTables.getTable('SmartDashboard')

#================================================================================================================================================================================

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.99.99.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

# Insert your processing code here
print("Connected!")


#================================================================================================================================================================================


#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Grab frame from video stream
    ret, frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    output_details = interpreter.get_output_details()
    #print (len (output_details))
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
   
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            if (object_name == 'Bola Azul'):
                xcenter = ((xmax - xmin)/2) 
                ycenter = ((ymax - ymin)/2)
            
                xmedio = (xmin + xcenter)
                ymedio = (ymin + ycenter)
                
                table.putNumber('xmedio', xmedio)
                table.putNumber('ymedio', ymedio)

                print (xmedio, ymedio)
            

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
