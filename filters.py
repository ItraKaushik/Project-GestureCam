import cv2
import numpy as np

def apply_sketch(img):
    # Convert the color frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image colors
    inverted = cv2.bitwise_not(gray)
    # Apply a strong blur to the inverted image
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Blend the grayscale image and inverted blur to create a sketch texture
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
    # Convert it back to 3 channels so it matches the webcam format
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_thermal(img):
    # Convert to grayscale first, then map it to a Jet (blue-to-red) thermal look
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def apply_sepia(img):
    # Transformation weight matrix for vintage sepia coloring
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    # Make sure values don't overflow past the 255 ceiling
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia
