import cv2,os
import numpy as np
from PIL import Image



recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        #faces=detector.detectMultiScale(imageNp)
        faces.append(imageNp)
        Ids.append(Id)

    return faces,Ids


faces,Ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(Ids))
recognizer.write('trainner2.yml')
cv2.waitKey(100000)
cv2.destroyAllWindows()




