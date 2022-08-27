# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
from http.client import HTTPException
import os
import app.FaceClassifier as FC
from typing import List

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import srsly
import uvicorn

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")

app = FastAPI(
    title="Hook Antrophometrics Module",
    version="1.0",
    description="Python API",
    openapi_prefix=prefix,
)

example_request = srsly.read_json("app/data/example_request.json")

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")

def read_image(bin_data, size=(224, 224)):
    """Load image

    Arguments:
        bin_data {bytes} --Image binary data

    Keyword Arguments:
        size {tuple} --Image size you want to resize(default: {(224, 224)})

    Returns:
        numpy.array --image
    """
    file_bytes = np.asarray(bytearray(bin_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def get_face_mesh(images):
    results = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for _, image in enumerate(images):
            # Convert the BGR image to RGB before processing.
            new_mesh = face_mesh.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ).multi_face_landmarks
            if new_mesh:
                results.append(new_mesh[0])
            else:
                raise HTTPException(status_code=418, detail="The image does not contain a face")
    return results

def get_highlight_landmarks(landmarks):
    highlight_landmarks = defaultdict(dict)
    highlight_landmarks["chin"] = landmarks[199]

    highlight_landmarks["mouth_right"] = landmarks[57]
    highlight_landmarks["mouth_left"] = landmarks[287]
    highlight_landmarks["mouth_bottom"] = landmarks[18]

    highlight_landmarks["between_eyes"] = landmarks[168]

    highlight_landmarks["nose_tip"] = landmarks[1]
    highlight_landmarks["nose_bottom"] = landmarks[2]
    highlight_landmarks["nose_right"] = landmarks[98]
    highlight_landmarks["nose_left"] = landmarks[327]

    highlight_landmarks["nose_bridge_right"] = landmarks[193]
    highlight_landmarks["nose_bridge_left"] = landmarks[417]

    highlight_landmarks["eyebrow_right_outer"] = landmarks[70]
    highlight_landmarks["eyebrow_right_inner"] = landmarks[107]

    highlight_landmarks["eyebrow_left_outer"] = landmarks[300]
    highlight_landmarks["eyebrow_left_inner"] = landmarks[336]

    highlight_landmarks["eye_right_outer"] = landmarks[226]
    highlight_landmarks["eye_right_inner"] = landmarks[243]
    highlight_landmarks["eye_right_bottom"] = landmarks[23]
    highlight_landmarks["eye_right_top"] = landmarks[27]

    highlight_landmarks["eye_left_outer"] = landmarks[446]
    highlight_landmarks["eye_left_inner"] = landmarks[463]
    highlight_landmarks["eye_left_bottom"] = landmarks[253]
    highlight_landmarks["eye_left_top"] = landmarks[257]

    return highlight_landmarks

def get_face(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes

def crop_faceBox(image, faceBox, padding=20):
    return image[
        max(0, faceBox[1] - padding) : min(faceBox[3] + padding, image.shape[0] - 1),
        max(0, faceBox[0] - padding) : min(faceBox[2] + padding, image.shape[1] - 1),
    ]

faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"
fairFaceModel = './models/res34_fair_align_multi_7_20190809.pt'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceNet = cv2.dnn.readNet(faceModel, faceProto)
faceClassifier = FC.FaceClassifier(fairFaceModel)

@app.post("/upload/antrophometrics")
async def upload(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        #Is file format valid
        if((file.content_type != "image/jpeg") & (file.content_type != "image/png")):
            raise HTTPException(status_code=422, detail="The file format is invalid")

        #Read the file
        try:
            contents = await file.read()
            image = read_image(contents)
            #If its png, convert it to jpeg
            if file.content_type == "image/png":
                cv2.imwrite('test.jpg', image)
                images.append(cv2.imread("test.jpg"))
            else:
                images.append(image)
                cv2.imwrite('test.jpg', image)
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            await file.close()

    # TODO: Get an average face_mesh using more than one image
    # TODO: Transform the coordinate system to be center on the nose_tip    
    face_mesh = get_face_mesh(images)[0]

    # TODO: Get an average age and gender using more than one image
    faceboxes = get_face(faceNet, images[0])
    face = crop_faceBox(images[0], faceboxes[0])
    cv2.imwrite('test/test.jpg',face)
    predictions = FC.format_prediction(faceClassifier.predict('test/')[0])

    # Create a json object to return
    landmarks = defaultdict(dict)

    for idx, landmark in enumerate(face_mesh.landmark):
        landmarks[idx]["x"] = landmark.x
        landmarks[idx]["y"] = landmark.y
        landmarks[idx]["z"] = landmark.z

    highlight_landmarks = get_highlight_landmarks(landmarks)

    age_results = predictions["Age"]
    gender_results = predictions["Gender"]
    race_results = predictions["Race"]

    return {
        "highlight_landmarks": highlight_landmarks,
        "landmarks": landmarks,
        "age": age_results,
        "gender": gender_results,
        "race": race_results
    }
