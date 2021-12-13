from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 64, 64

from tensorflow.keras.optimizers import Adam

# load the model we saved
model = load_model('model/')

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# predicting images

img = image.load_img('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/00000.ppm', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)

# predicting multiple images at once
# img = image.load_img('test2.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)
#
# pass the list of multiple images np.vstack()
# images = np.vstack([x, y])
# classes = model.predict_classes(images, batch_size=10)

# print the classes, the images belong to
