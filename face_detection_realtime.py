import cv2
from random import randrange

# Build the classifier
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# Capture video from webcam
webcam = cv2.VideoCapture(3, cv2.CAP_DSHOW)

# Iterate over video frames
while True:
    # Read the frame
    successful_frame_read, frame = webcam.read()

    # Convert the frame to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Open CV is BGR not RGB

    # Detect the face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame) 

    # Draw rectangles around faces
    for (x,y,w,h) in face_coordinates:       # x = x-coordinate, y = y-coordinate, w = width, h = height
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # Display the frame
    cv2.imshow("THE Face Detector",frame)

    # Keeps the image open until a key is pressed
    key = cv2.waitKey(1)      # 1 = Frame changes every milisecond
    
    # Press q to quit video feed
    if key==81 or key==113:
        break
    #cv2.destroyAllWindows()

# Release the VideoCapture object
webcam.release()
