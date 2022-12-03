#https://robotpy.readthedocs.io/projects/pynetworktables/en/stable/examples.html#pynetworktables-examples
import cv2
import threading
import time
from networktables import NetworkTables

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

#Extrair Dashboard
table = NetworkTables.getTable('SmartDashboard')

table.setDefaultNumber('Megazord', 0)

table.putNumber('Megazord', 7563)
Num = 7563

while True:
	TP = table.getNumber('axe', 0)
	table.putNumber('Megazord', 7563)
	print (TP)
	time.sleep(1)
