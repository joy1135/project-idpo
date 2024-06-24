import keras
from PIL import Image
import numpy as np

class_name = ['0', '1']

model = keras.models.load_model('the_integrity_of_the_box.keras')

img = Image.open('1.jpg')

img = img.resize((512,512))
img_array = np.array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
predict = model.predict(img_array)
predict_class_index = np.argmax(predict, axis=1)[0]
predict_class_name = class_name[predict_class_index]




print(predict_class_name)


 