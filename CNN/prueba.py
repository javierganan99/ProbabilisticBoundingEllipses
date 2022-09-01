import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# IMPORTANT!!!
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

base_dir = "dataset_frames/"
train_data_dir = os.path.join(base_dir, "train")
validation_data_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

model = tf.keras.models.Sequential(
    [
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(100, 100, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    brightness_range=[0,1.5],
    rescale=1 / 255
)

# datagen = ImageDataGenerator(width_shift_range=[-100, 100],          
#                     height_shift_range=[-100, 100], 
#                     rotation_range=120, brightness_range=[0.2, 1.5],
#                     zoom_range = [0.3, 1.5] ,shear_range=50)

pic = train_datagen.flow_from_directory(
    train_data_dir, target_size=(100, 100), batch_size=1, class_mode="binary"
)
# pic = train_datagen.flow(img_tensor, batch_size =1)
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))#Plots our figures
for j in range(1000):
    for i in range(1,17):
       plt.subplot(4, 4, i)
       batch = pic.next()
       image_ = batch[0][0]#.astype(‘uint8’)
       plt.imshow(image_)
    plt.show()