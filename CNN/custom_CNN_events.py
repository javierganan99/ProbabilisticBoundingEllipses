import os 
import tensorflow as tf
# IMPORTANT!!!
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import RMSprop

base_dir = 'dataset/'
train_data_dir = os.path.join(base_dir, 'train')
#validation_data_dir = os.path.join(base_dir, 'val')
#test_dir = os.path.join(base_dir, 'test')
model = tf.keras.models.Sequential([
# Note the input shape is the desired size of the image 200x200 with 3 bytes color
# This is the first convolution
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
# The second convolution
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The third convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The fourth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# # The fifth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# Flatten the results to feed into a DNN
tf.keras.layers.Flatten(),
# 512 neuron hidden layer
tf.keras.layers.Dense(512, activation='relu'),
# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255)
#validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,  
    target_size=(100,100), 
    batch_size=8,
    class_mode='binary')

# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir, 
#     target_size=(100,100), 
#     batch_size=5,
#     class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=None,  
    epochs=10,
    verbose=1,
    validation_data = None, #validation_generator
    validation_steps=None)

#print(model.evaluate(validation_generator))
model.save('CUSTOM_Model_events.h5')

