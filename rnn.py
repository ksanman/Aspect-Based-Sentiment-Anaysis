import tflearn as tf
import numpy as np
from tensorflow import reset_default_graph
class RNN:
    """ Recurrent neural network model. """
    def __init__(self):
        pass

    def build_model(self, x, y, x_val, y_val):
        reset_default_graph()
        net = tf.input_data(shape=(None, x.shape[1]))
        net = tf.embedding(net, input_dim=1000, output_dim=128)
        net = tf.lstm(net, 128, dropout=0.8)
        net = tf.fully_connected(net, 2, activation='softmax')
        net = tf.regression(net, learning_rate=0.000001)
        self.model = tf.DNN(net)
        self.model.fit(x, y, n_epoch=100, batch_size=1000, show_metric=True, validation_set=(x_val, y_val))
       
    def score(self, X, Y):
        results = []
        for i, x in enumerate(X):
	    y_0 = self.model.predict([x])
            results.append(True if np.argmax(y_0[0]) == Y[i] else False)
        accuracy = 0
	for r in results:
            
            if r == True:
                accuracy += 1
       
        return float(accuracy)/float(len(Y))

