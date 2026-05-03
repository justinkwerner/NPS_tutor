#########################################################
# NPS Payload Simulator
# Simple payload simulator designed to use UDP coms
# 
# Author: Alex Savattone
# Date: July 13 2022
#########################################################

import socket
import json
import sys
import logging
from gpiozero import LED
from datetime import datetime

# setup logging
file_handler = logging.FileHandler(filename='main.log') # log to file
stdout_handler = logging.StreamHandler(sys.stdout) # print to console
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

# define gpio pin for led and inital state
led = LED(4) 
led.off()

HOST = ''  # binds with all avaliable interfaces
PORT = 8080 # Port to listen on (non-privileged ports are > 1023)

try:
    # create udp socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # bind socket to ip and port
        s.bind((HOST, PORT))
        while True:
            logging.info("Server is listening")
            
            data, addr = s.recvfrom(1024) # wait for data, this is blocking
            logging.info(f"Cmd recieved from {addr}")
            
            msg = json.loads(data) # parse json object to python dictionary 
            logging.info(msg)
            
            cmd = msg['cmd']
            
            if cmd == 'sq':
                if msg['value'] in range(0,10):
                    r = msg['value'] * msg['value']
                else:
                    r = -1
                
            elif cmd == 'led_on':
                logging.debug("Led Pyload On")
                r = 1
                led.on()
            
            elif cmd == 'led_blink':
                logging.debug("Led Payload Blink")
                r = 1
                led.blink()
            
            elif cmd == 'led_off':
                logging.debug("LED Payload Off")
                r = 0
                led.off()
            
            elif cmd == 'custom':
                var1 = msg['var1']
                var2 = msg['var2']
                logging.debug(f"cmd={cmd}, var1={var1}, var2{var2}")
                r = 0
                
            else:
                r = -1
            
            sts = int(led.is_lit) # read led pin status
            ts = int(datetime.now().timestamp()) # create timestamp
            return_msg = {"value":r, "ts":ts, "led_status":sts} # build return message dictionary
            logging.info(return_msg)
            return_msg = json.dumps(return_msg) # convert python dictionary to json
            s.sendto(return_msg.encode("utf-8"), addr) # sends return message to original ip
            
# catch key board inturupt and exit program
except KeyboardInterrupt:
    logging.info("KeyboardInterrupt, exiting program")
    sys.exit()
    
# catch unexpected errors and print traceback
except:
    logging.exception("Error!", exc_info=True)

# last code that always runs, cleanup gpio resource
finally:
    led.close()
    
