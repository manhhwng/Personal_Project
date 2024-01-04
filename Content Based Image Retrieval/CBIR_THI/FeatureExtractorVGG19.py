# Import the libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


count = 0 
if __name__ == '__main__':
    fe = FeatureExtractor()
    for img_path in sorted(Path("./static/database").glob("*.jpg")):
        count+=1 
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/FeaturesVGG19") / (img_path.stem + ".npy")  
        np.save(feature_path, feature)
    print('Number of images:', count)