from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences: list, dimension=10000):
    """Method to vectorize data.

    Args:
        sequences (array): An array of newswires as lists of word indices
        dimension (int, optional): Number of words occuring in data. Defaults to 10000.

    Returns:
        array: Array containing vectorized newswires.
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def to_one_hot(labels: list, dimension=46):
    """Method to one hot encode labels in data.

    Args:
        labels (list): Array of labels in data
        dimension (int, optional): Number of topics of news in data. Defaults to 46.

    Returns:
        array: Array of one hot encoded labels.
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# Load Reuters-dataset, limit number of words to 10 000
(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

# Vectorize train and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# Define neural network model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(46, activation='softmax'))

# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# Set aside validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Train model with 15 epochs
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=15,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Plot loss and accuracy of model
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Show results of model
results = model.evaluate(x_test, one_hot_test_labels)
print(model.metrics_names)
print(results)

# Make predictions for test data, print most propable topic for 10 instances
predictions = model.predict(x_test)
for i in range(10):
    print('Topic: ' + str(np.argmax(predictions[i])) + ' Probability: ' + str(predictions[i][np.argmax(predictions[i])]))