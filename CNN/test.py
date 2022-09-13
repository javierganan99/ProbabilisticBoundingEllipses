import keras
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tensorflow as tf
# IMPORTANT!!!
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# It loads the model
model = keras.models.load_model('/path_to_the_model/model.h5')

base_dir = '/path_to_the_directeory_containing_test_folder/'
test_dir = os.path.join(base_dir, 'test')

test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 1, class_mode = 'binary', target_size = (100, 100))

cont = 0
total = 0
while total < len(test_generator):
    image = test_generator[total]
    pred = model.predict(image[0])
    if (pred >= 0.5 and image[1] == 1) or (pred < 0.5 and image[1] == 0):
        cont += 1
    total += 1
    
print("Accuracy over the test set is " + str(cont/total * 100))