# reuters-classifier

This easy project was made to get some understanding on neural network architecture and to get some insight about parameters that have an effect on a NN model. First model was trained with an architecture as following:

`model = models.Sequential()`
`model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))`
`model.add(layers.Dense(64, activation='relu'))`
`model.add(layers.Dense(46, activation='softmax'))`

and with 20 epochs. After plotting was noticed that model started overfitting after 9-10 epochs, so the amount of epochs was reduced to 9. After next run a loss of 0.98 and accuracy of ~79% was achieved. 