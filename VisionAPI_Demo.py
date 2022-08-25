import io


import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'research-project-360323-4c28b9bddffc.json'

client = vision.ImageAnnotatorClient()

FILE_NAME = 'upload.png'
FOLDER_PATH = 'C:/Users/Henri Van Oost/Documents/MCT/Semester5/Research Project/Henri/upload/'

with io.open(os.path.join(FOLDER_PATH, FILE_NAME), 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations
df = pd.DataFrame(columns=['locale', 'description'])

for text in texts:
    df = df.append(
        dict(locale=text.locale, description=text.description), ignore_index=True)
print(df['description'][0])


# print(client.text_detection(image=image))
