import time 

import cv2
import mss
import numpy 

from grabscreen import Record

def main():
    while True:
        img = Record()
        cv2.imshow('game window', img)

if __name__ == '__main__':
    main()
