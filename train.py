# 1. Import Library
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
import matplotlib.pylab as plt

# 2. Generate a Dataset
image_size = (224, 224)
batch_size = 64
num_classes = 39

train_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
train_generator = train_datagen.flow_from_directory(
    "data/train",
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
    shuffle=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    "data/test",
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
    shuffle=False
)


# 3. Build model

def make_model():
    # create base model
    base_model = MobileNet(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    # create main model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    # Add some new fully connected layers to
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    models = Model(inputs=inputs, outputs=x, name="Plant_Disease_MobileNet")
    return models


model = make_model()
model.summary()
keras.utils.plot_model(model, show_shapes=True)

# 4. Train Model
epochs = 4
learning_rate = 0.001
decay_rate = learning_rate / epochs

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")]
)


def scheduler(epoch, lr):
    if epoch < 5:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))


callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/checkpoint-{epoch}-{val_acc}.hdf5',
        save_best_only=True,
        monitor='val_acc',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(monitor='val_loss',
                                  patience=3,
                                  restore_best_weights=True),
    keras.callbacks.LearningRateScheduler(scheduler)
]

history = model.fit(train_generator,
                    validation_data=train_generator,
                    callbacks=callbacks,
                    epochs=epochs,
                    steps_per_epoch=30,
                    validation_steps=30)

# 5. Evaluate Model

# returns loss and metrics
# loss, acc = model.evaluate(test_generator)
# print("loss: %.2f" % loss)
# print("acc: %.2f" % acc)

# plot val,acc history
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid(False)
plt.xlabel('Epochs')
plt.ylabel('Loss Magnitude')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.grid(False)
plt.xlabel('Epochs')
plt.ylabel('Loss Magnitude')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
# confusion matrix
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
class_labels = list(test_generator.class_indices.keys())
print(class_labels)
print(confusion_matrix(test_generator.classes, y_pred, normalize='all'))
print(classification_report(test_generator.classes, y_pred, target_names=class_labels))


