import cv2 as cv

""" Identification """

faceProto = "/models/opencv_face_detector.pbtxt"
faceModel = "/models/opencv_face_detector_uint8.pb"

ageProto = "/models/age_deploy.prototxt"
ageModel = "/models/age_net.caffemodel"

genderProto = "/models/gender_deploy.prototxt"
genderModel = "/models/gender_net.caffemodel"

faceNet=cv.dnn.readNet(faceModel, faceProto)
ageNet=cv.dnn.readNet(ageModel,ageProto)
genderNet=cv.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
confidence_threshold = 0.9
padding=20

""" Face highliting """

def faceBox(faceNet, frames):
    frameHeight=frames.shape[0]
    frameWidth=frames.shape[1]
    blob=cv.dnn.blobFromImage(frames, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv.rectangle(frames, (x1,y1),(x2,y2),(0,255,0), 1)
    return frames, bboxs

""" Video display """

def DisplayVid():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('testvideo', fourcc, 20.0, (640, 480))

    while (True):
        ret, frame = cap.read()
        frameFace, bboxes = faceBox(faceNet, frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        out.write(frame)

        for bbox in bboxes:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = "{},{}".format(gender, age)
            cv.rectangle(frameFace, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv.LINE_AA)
        cv.imshow("Age-Gender", frameFace)
        k = cv.waitKey(1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if not (cap.isOpened()):
            print("Could not open video device")

    cap.release()
    out.release()
    cv.destroyAllWindows()

DisplayVid()