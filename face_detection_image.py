import cv2
from random import randrange

# Build the classifier
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# Import the image for detection
img = cv2.imread("two_kids.jpg")

# Convert the image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Open CV is BGR not RGB

# Detect the face coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) 

# Draw rectangles around faces
for (x,y,w,h) in face_coordinates:       # x = x-coordinate, y = y-coordinate, w = width, h = height
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(200,256), randrange(200,256), randrange(200,256)), 2)  # 2 = Thickness of the rectangle

# Draw rectangles for individual face
"""(x,y,w,h)=face_coordinates[0]
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
(x,y,w,h)=face_coordinates[1]
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)"""

# Display the image
cv2.imshow("THE Face Detector",img)

# Keeps the image open until a key is pressed
cv2.waitKey()
