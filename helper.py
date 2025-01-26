import cv2
import numpy as np
code ={
      'angry':0,
      'disgust': 1,
      'fear': 2,
      'happy': 3,
      'neutral': 4,
      'sad': 5,
      'surprise': 6
      }

def getcode(n) :
    for x , y in code.items() :
        if n == y :
            return x
def image_preprocessing(img) :
    img = cv2.resize(img,(48,48))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(equalized, (5, 5), 0)
    normalized = denoised / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    return normalized