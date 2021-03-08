# reuters-classifier

This easy project was made to get some understanding on neural network architecture and to get some insight about parameters that have an effect on a NN model. First model was trained with an architecture as following:

`model = models.Sequential()`  
`model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))`  
`model.add(layers.Dense(64, activation='relu'))`  
`model.add(layers.Dense(46, activation='softmax'))`  

and with 20 epochs. After plotting was noticed that model started overfitting after 9-10 epochs, so the amount of epochs was reduced to 9. After next run a loss of 0.98 and accuracy of ~79.7% was achieved. 

After that the network architecture was changed to

`model = models.Sequential()`  
`model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))`
`model.add(layers.Dropout(0.5))`
`model.add(layers.Dense(128, activation='relu'))`
`model.add(layers.Dropout(0.5))`  
`model.add(layers.Dense(46, activation='softmax'))`

and was trained with 20 epochs. Overfitting started at around 15th epoch so run with 15 epochs followed and resulted with loss of 1.05 and accuracy of 79.3%. After experimenting  with more layers, and bigger/smaller layers was noticed that in this case it was difficult to obtain better results than with the simplest approach tested first. The next step would be adding regularization to network or obtaining more data to crunch (which is not an option with this ready made dataset).