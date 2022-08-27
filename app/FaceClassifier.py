import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms

torch_device = torch.device('cpu')

class NotAFaceException(Exception):
    def __init__(self, message="No faces found in the image"):
        self.message = message

classifier_Lists = {
    "RacesList": ["White", "Black", "Latino_Hispanic", "Asian", "Asian", "Indian", "Middle Eastern"],
    "GenderList": ["Male", "Female"],
    "AgeList": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def clear_directory(directory):
    os.system('rm -fr ' + directory + '*')

class FaceClassifier():
    model = []
    def __init__(self, path):
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(
            torch.load(
                path,
                map_location=torch_device
            )
        )
        self.model = self.model.to(torch_device)
        self.model.eval()

    def read_image(self, path):
        image = Image.open(path)
        image = transform(image)
        image = image.view(1, 3, 224, 224)
        image = image.to(torch_device)
        
        return image

    def predict(self, imgs_path = 'detected_faces/'):
        '''
        Given an image of a face, predicts the race, gender and age

        Parameters:
        imgs_path: Path where the images are saved

        Outputs:
        list of dictionaries with predictions for every image
        '''
        img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]

        results = []
        for img_name in img_names:
            image = self.read_image(img_name)

            outputs = self.model(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
        
            race_outputs = outputs[:7]
            gender_outputs = outputs[7:9]
            age_outputs = outputs[9:18]

            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
            age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

            prediction = {
                "Image": img_name,
                "Race": race_score,
                "Gender": gender_score,
                "Age": age_score
            }

            results.append(prediction)
        
        return results

def show_predictions(predictions):
    for prediction in predictions:
        print(
            "Image: ", prediction["Image"], '\n',
            "Race: ", classifier_Lists["RacesList"][np.argmax(prediction["Race"])], '\n',
            "Gender: ", classifier_Lists["GenderList"][np.argmax(prediction["Gender"])], '\n', 
            "Age: ", classifier_Lists["AgeList"][np.argmax(prediction["Age"])], '\n'
        )

def format_prediction(prediction):
    return {
        "Image": prediction["Image"],
        "Race": classifier_Lists["RacesList"][np.argmax(prediction["Race"])],
        "Gender": classifier_Lists["GenderList"][np.argmax(prediction["Gender"])],
        "Age": classifier_Lists["AgeList"][np.argmax(prediction["Age"])],
    }

if __name__ == "__main__":
    classifier = FaceClassifier('models/')

    #detector.detect_faces_in_folder('test/', SAVE_AT='detected_faces/')
    results = classifier.predict()

    show_predictions(results)


'''
@inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
'''