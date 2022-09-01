import keras
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model('CUSTOM_Model_events_100_100.h5')

base_dir = 'dataset/CNN_events'
test_dir = os.path.join(base_dir, 'test')

test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 1, class_mode = 'binary', target_size = (100, 100))

cont = 0
total = 0
while total < len(test_generator):
    image = test_generator[total]
    pred = model.predict(image[0])
    if (pred[0][1] >= 0.5 and image[1] == 1) or (pred[0][1] < 0.5 and image[1] == 0):
        cont += 1
    print(image[1])
    total += 1
    print(pred)
    
print("El rendimiento sobre el conjunto de test es: " + cont/total * 100)