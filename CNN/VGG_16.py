import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16

base_dir = "/home/grvc/GitHub/wfilter-cnn/dataset/CNN_events"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Directory with our training person pictures
train_persons_dir = os.path.join(train_dir, "ONLY_Persons")
# Directory with our training NO person pictures
train_nopersons_dir = os.path.join(train_dir, "croppedNoPersons")
validation_persons_dir = os.path.join(validation_dir, "ONLY_Persons")
validation_nopersons_dir = os.path.join(validation_dir, "croppedNoPersons")
test_persons_dir = os.path.join(test_dir, "ONLY_Persons")
test_nopersons_dir = os.path.join(test_dir, "croppedNoPersons")

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=20, class_mode="binary", target_size=(100, 100)
)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = val_datagen.flow_from_directory(
    validation_dir, batch_size=20, class_mode="binary", target_size=(100, 100)
)

base_model = VGG16(
    input_shape=(100, 100, 3),  # Shape of our images
    include_top=False,  # Leave out the last fully connected layer
    weights="imagenet",
)

for layer in base_model.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation="relu")(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["acc"])

vgghist = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=100, epochs=10)

model.save("my_model_VGG_16_100_100_events.h5")
