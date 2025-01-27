import cv2 as cv
import cv2.data as cd



# Initialize the webcam
cam = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    status, img = cam.read()
    if not status:
        print("Failed to capture image")
        break
    
    # Convert the frame to grayscale
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Load the cat face classifier
    classifier = cv.CascadeClassifier(cd.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect cat faces
    faces = classifier.detectMultiScale(img2, 1.3, 5)
    
    # Draw rectangles around detected faces
    for face in faces:
        x=face[0]
        y=face[1]
        h=face[2]
        w=face[3]
        xl = x + w
        yl = y + h
        cv.rectangle(img, (x, y), (xl, yl), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv.imshow("my image", img)
    
    # Break the loop when 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the webcam and close windows
cam.release()
cv.destroyAllWindows()
