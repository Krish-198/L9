import cv2,os,numpy
har="haarcascade_frontalface_default.xml"
data="Data_sets"
print("Recognizing face please be in sufficient light")
(images,labels,names,id) = ([],[],{},0)
for(subdir,dir,file) in os.walk(data):
    for subdir in dir:
        names[id]=subdir
        path=os.path.join(data,subdir)
        for filename in os.listdir(path):
            pp=path+'/'+filename
            label=id
            images.append(cv2.imread(pp,0))
            labels.append(int(label))
        id+=1
(width,height) = (130,100)
(images,labels) = [numpy.array(i) for i in [images,labels]]
r=cv2.face.LBPHFaceRecognizer_create()
r.train(images,labels)
face=cv2.CascadeClassifier(har)
print("Please turn on your camera ")
webcam=cv2.VideoCapture(0)


while  True:
    (c,d)=webcam.read()
    gray=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(d,(x,y),(x+w,y+h),(0,0,255),3)
        g=gray[y:y+h,x:x+w]
        faceresize=cv2.resize(g,(width,height))
        detect=r.predict(faceresize)
        cv2.rectangle(d,(x,y),(x+w,y+h),(0,0,255),3) 
        if detect[1]:
            cv2.putText(d,(names[detect[0]],detect[1]))
            


    cv2.imshow("Krish",d)
