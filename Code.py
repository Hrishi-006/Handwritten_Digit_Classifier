import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(data, labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data() #Loading MNIST dataset 
data = data.astype('float32') / 255.0   # Normalizing the data for more accuracy
test_data = test_data.astype('float32') / 255.0


data = data.reshape(-1, 28*28)    #Turning each 28x28 pixel image into a 1D array of 784 pixels
test_data = test_data.reshape(-1, 28*28)


max_examples = 10000    #Limit test dataset training to 10000 expamples
data = data[:max_examples]
labels = labels[:max_examples]


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(units=10, activation='softmax')     #Single dense layer neural network with softmax activation being used to calculate probabilities
])
model.compile(optimizer='adam',                           #adam algorithm is used for updating weights for more accuracy
              loss='sparse_categorical_crossentropy',     #sum of negative log of True probability* models preicted probability
              metrics=['accuracy'])
model.fit(data, labels, epochs=50, batch_size=100)      #Training the model. Epochs is the number of times we pass the whole dataset. Batch_size is the number of samples per gradient update



loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")    #Accuracy obtained is 91.54% and loss is 0.3041


test_index = 9  # To check with the test dataset
true_label = test_labels[test_index]
number_to_predict = np.expand_dims(test_data[test_index], axis=0)   #Converts test data into a 1D array
prediction = model.predict(number_to_predict)
predicted_label = np.argmax(prediction) #Finds the index with the highest probability
print(f"Prediction: {predicted_label}, True Label: {true_label}")

