"""
training covnet model on rgb images with no transform
"""

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

"""
import pathlib
relative_filepath = "MICC-F220_No_Transform"
absolute_filepath = os.path.dirname(os.path.realpath(relative_filepath))
data_dir = tf.keras.utils.get_file(fname=absolute_filepath)
data_dir = pathlib.Path(data_dir)
"""

# getting path to images
import pathlib

relative_filepath = "rgb"
#absolute_filepath = os.path.dirname(os.path.realpath(relative_filepath))
#absolute_filepath = os.path.normpath(os.path.join(absolute_filepath, relative_filepath))
absolute_filepath = os.path.abspath(relative_filepath)
print("absolute_filepath = ", absolute_filepath)


# defining parameters
batch_size = 32
img_height = 224
img_width = 224

# creating training dataset
# using 80% data for training and 20% for validation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    absolute_filepath,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# creating validation dataset
# using 80% data for training and 20% for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    absolute_filepath,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# creating classnames
class_names = train_ds.class_names
print(class_names)

'''
# visualizing data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
'''

# checking data shape in training dataset
# output will be in the form (batch_size, img_height, img_width, channels)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# checking data shape in validation dataset
for image_batch, labels_batch in val_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# standardizing the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Printing the min and max values in normalized image indexed 0.
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

"""
 Configure the dataset for performance
 Using buffered prefetching so we can yield data from disk 
 without having I/O become blocking. These are two 
 important methods you should use when loading data.
 .cache() keeps the images in memory after they're loaded off disk during the first epoch. 
 This will ensure the dataset does not become a bottleneck while training your model. 
 If your dataset is too large to fit into memory, 
 you can also use this method to create a performant on-disk cache.
 .prefetch() overlaps data preprocessing and model execution while training.
"""
#AUTOTUNE = tf.data.experimental.AUTOTUNE
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#train_ds = train_ds.cache().take(k).repeat()
#val_ds = val_ds.cache().take(k).repeat()

# creating the model
num_classes = 2
model = tf.keras.Sequential(
    [
#        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, activation="relu", input_shape=(img_height,img_width,3)),
#        layers.Conv2D(32,3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.GlobalAveragePooling2D(),
       # layers.Flatten(),
        layers.Dense(num_classes)
    ]
)
#model.build((224,224,3))
#model.summary()


# compiling the model
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


#training the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=8
)

model.summary()


#evaluate the model
plt.title("rgb no transform")
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.savefig("rgb_no_transform_acc_vs_epoch.jpg")
#plt.show()

#test_loss, test_acc = model.evaluate(train_ds,  val_ds, verbose=2)