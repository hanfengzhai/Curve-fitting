import pyDOE, sys, time, tqdm, os
import tensorflow_probability as tfp
from tensorflow.keras.layers import InputLayer, Dense, Layer
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from visualization_utils import plot, init_color_object
colors = init_color_object()


class NeuralNet:
    # Initialize the class
    def __init__(self, t_train, x_train, t_test, x_test, layers, t_min, t_max, option='sin'):
        self.t_train = t_train
        self.x_train = x_train
        self.t_test  = t_test
        self.x_test  = x_test
        self.loss_history = {"train": [], "test": []}

        # Initialize NNs with given number of layers and domain bounds [t_min, tmax]
        self.t_min = t_min
        self.t_max = t_max
        self.initialize_nn(layers)
        
        self.option = option

    def xavier_init(self, size):
        in_dim  = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def initialize_nn(self, layers):
        """Functions used to establish the initial neural network"""
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        self.weights = weights
        self.biases = biases
        self.trainable_variables = self.weights + self.biases

    def net(self, X):
        H = 2.0 * (X - self.t_min) / (self.t_max - self.t_min) - 1.0
        for l in range(0, len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            if self.option == 'sin': H = tf.sin(tf.add(tf.matmul(H, W), b))
            elif self.option == 'relu': H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            elif self.option == 'tanh': H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    @tf.function
    def loss_train(self):
        x_pred = self.net(self.t_train)
        loss = tf.reduce_mean(tf.square(self.x_train - x_pred)) * 10
        return loss

    @tf.function
    def loss_test(self):
        x_pred = self.net(self.t_test)
        loss = tf.reduce_mean(tf.square(self.x_test - x_pred)) * 10
        return loss

    def get_test_error(self):
        x_pred = self.net(self.t_test)
        error_x = tf.norm(self.x_test - x_pred, 2) / tf.norm(self.x_test, 2)
        return error_x

    def train(self, nIter: int, learning_rate: float, idxOpt: int):
      """Function used for training the model"""
      if idxOpt == 1:
          # mode 1: running the Gradient Descent optimization
          self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
      elif idxOpt == 2:
          # mode 2: running the Adam optimization
          self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
      else:
          raise ValueError("Unsupported idxOpt")

      varlist = self.trainable_variables
      start_time = time.time()

      for it in tqdm.tqdm(range(nIter)):
          with tf.GradientTape() as tape:
              loss_value = self.loss_train()

          # Compute gradients
          gradients = tape.gradient(loss_value, varlist)

          # Apply gradients
          self.optimizer.apply_gradients(zip(gradients, varlist))

          # Print training progress
          if it % 500 == 0:
              elapsed = time.time() - start_time
              loss_train = loss_value.numpy()
              loss_test = self.loss_test().numpy()
              
              self.loss_history["train"].append(loss_train)
              self.loss_history["test"].append(loss_test)
              tqdm.tqdm.write('It: %d, Train Loss: %.3e, Test Loss: %.3e, Time: %.2f' %
                                (it, loss_train, loss_test, elapsed))
              start_time = time.time()


class TrainNeuralNet:
    def __init__(self, n_total=None, n_train=None, lr=0.01, epoch=2000, 
                 actv_func='sin', output_dir=None):
        self.n_total    = n_total
        self.n_train    = n_train
        self.lr         = lr
        self.epoch      = epoch
        self.actv_func  = actv_func
        self.output_dir = output_dir

        # fix random seed within the class
        np.random.seed(123); tf.random.set_seed(123)

        print(f'{colors.BLUE}n_total:{colors.RESET} {n_total}\n{colors.BLUE}n_train:{colors.RESET} {n_train}')

    def prepare_data(self):

        self.t_all = np.linspace(-1, 1, self.n_total)
        self.x_all = x = np.sin(5 * self.t_all)

        train_indices = np.zeros(self.t_all.size, dtype=bool)
        train_indices[:self.n_train] = True
        np.random.shuffle(train_indices)

        self.t_train = self.t_all[train_indices]
        self.x_train = self.x_all[train_indices]
        self.t_train = tf.reshape(tf.cast(self.t_train, dtype = tf.float32), shape=(-1, 1))
        self.x_train = tf.reshape(tf.cast(self.x_train, dtype = tf.float32), shape=(-1, 1))

        self.t_test = self.t_all[~train_indices]
        self.x_test = self.x_all[~train_indices]
        self.t_test = tf.reshape(tf.cast(self.t_test, dtype = tf.float32), shape=(-1, 1))
        self.x_test = tf.reshape(tf.cast(self.x_test, dtype = tf.float32), shape=(-1, 1))

    def train_model(self):
        self.model = NeuralNet(t_train=self.t_train, x_train=self.x_train, t_test=self.t_test, x_test=self.x_test,
                    layers=[1, 100, 100, 100, 1], t_min=self.t_all.min(0), t_max=self.t_all.max(0), option=self.actv_func)

        start_time = time.time()
        self.model.train(self.epoch, learning_rate=self.lr, idxOpt=1)
        elapsed = time.time() - start_time
        print(f'{colors.BLUE}Training time:{colors.RESET} %.4f' % (elapsed))

        self.pred_all = self.model.net(tf.reshape(tf.cast(self.t_all, dtype=tf.float32), shape=(-1, 1))).numpy().flatten()
        print(f'{colors.BLUE}Norm of Differnece:{colors.RESET} %e' % (self.model.get_test_error().numpy()))

    def post_process(self, option='lr'):
        r2_train = r2_score(self.x_train, self.model.net(self.t_train).numpy())
        r2_test = r2_score(self.x_test, self.model.net(self.t_test).numpy())
        print(f'{colors.BLUE}R2 train:{colors.RESET} {r2_train:.4f}')
        print(f'{colors.BLUE}R2 test:{colors.RESET} {r2_test:.4f}')

        if option=='lr': # task 1
            os.makedirs(f'{self.output_dir}/lr', exist_ok=True)
            plot(self.t_all, self.x_all, self.pred_all, self.t_train, self.x_train, self.model, 
                title=rf"Gradient Descent Optimization w/ $R^2$ = {r2_train:.2f} & {r2_test:.2f}", 
                vars=self.lr, output_dir=f'{self.output_dir}/lr')
        elif option=='train_ratio': # task 2
            os.makedirs(f'{self.output_dir}/train_ratio', exist_ok=True)
            plot(self.t_all, self.x_all, self.pred_all, self.t_train, self.x_train, self.model, 
                title=rf"Gradient Descent Optimization w/ $R^2$ = {r2_train:.2f} & {r2_test:.2f}", 
                vars=self.n_train, output_dir=f'{self.output_dir}/train_ratio')
        elif option=='activation':
            os.makedirs(f'{self.output_dir}/activation', exist_ok=True)
            plot(self.t_all, self.x_all, self.pred_all, self.t_train, self.x_train, self.model, 
                title=rf"Gradient Descent Optimization w/ $R^2$ = {r2_train:.2f} & {r2_test:.2f}", 
                vars=None, output_dir=f'{self.output_dir}/activation')
        
    def print_info(self):
        print('############################################################')
        print('#                   Training Finished!                     #')
        print('############################################################')