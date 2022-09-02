import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
# IMPORTANT!!!
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

base_dir = "dataset_frames/"
train_data_dir = os.path.join(base_dir, "train")
validation_data_dir = os.path.join(base_dir, "val")

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
    brightness_range=[0.2,1.5],
    rescale=1 / 255
)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(100, 100), batch_size=20, class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, target_size=(100, 100), batch_size=5, class_mode="binary"
)


earlyStopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min")

mcp_save = ModelCheckpoint(
    "Best_Model_frames.h5", save_best_only=True, monitor="val_loss", mode="min"
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode="min"
)

history = model.fit_generator(
    train_generator,
    epochs=30,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
)



# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy [%]', fontsize = 30)
plt.xlabel('Epochs', fontsize = 30)
plt.legend(['train', 'validation'], loc='upper left', fontsize = 20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss', fontsize = 30)
plt.xlabel('Epochs', fontsize = 30)
plt.legend(['train', 'validation'], loc='upper left', fontsize = 20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
plt.show()



#print(model.evaluate(validation_generator))

model.save("Final_Model_frames.h5")
