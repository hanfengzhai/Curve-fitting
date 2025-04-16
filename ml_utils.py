class NeuralNet:
    # Initialize the class
    def __init__(self, t_train, x_train, t_test, x_test, layers, t_min, t_max):
        self.t_train = t_train
        self.x_train = x_train
        self.t_test = t_test
        self.x_test = x_test
        self.loss_history = {"train": [], "test": []}

        # Initialize NNs with given number of layers and domain bounds [t_min, tmax]
        self.t_min = t_min
        self.t_max = t_max
        self.initialize_nn(layers)

    def xavier_init(self, size):
        in_dim = size[0]
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
            H = tf.sin(tf.add(tf.matmul(H, W), b))
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
          if it % 100 == 0:
              elapsed = time.time() - start_time
              loss_train = loss_value.numpy()
              loss_test = self.loss_test().numpy()
              self.loss_history["train"].append(loss_train)
              self.loss_history["test"].append(loss_test)
              tqdm.tqdm.write('It: %d, Train Loss: %.3e, Test Loss: %.3e, Time: %.2f' %
                                (it, loss_train, loss_test, elapsed))
              start_time = time.time()


def plot(t_all, x_all, pred_all, t_train, x_train, title: str):
    fig = plt.figure(figsize=(21, 24))

    ax = plt.subplot(311)
    ax.plot(t_all, x_all, "r", label="Exact Solution")
    ax.plot(t_all, pred_all, "b--", label="Prediction")
    ax.scatter(t_train, x_train, s=30, c="g", label="Training Data Point")
    ax.legend()
    ax.set_xlabel("$t$", fontsize = 15)
    ax.set_ylabel("$x$", fontsize = 15, rotation = 0)
    ax.set_title("$Fitting$", fontsize = 15)

    ax = plt.subplot(312)
    ax.plot(t_all, pred_all - x_all, "b-")
    ax.set_xlabel("$t$", fontsize = 15)
    ax.set_ylabel("Prediction - Exact Solution", fontsize = 15, rotation = 90)
    ax.set_title("Difference between Prediction and Exact Solution", fontsize = 15)

    ax = plt.subplot(313)
    loss_train = np.array(model.loss_history["train"])
    loss_test = np.array(model.loss_history["test"])
    ax.scatter(np.arange(loss_train.size) * 100, loss_train, s=30, marker="x", label="Train Loss")
    ax.scatter(np.arange(loss_train.size) * 100, loss_test, s=30, marker="+", label="Test Loss")
    ax.legend()
    ax.set_xlabel("$iterations$", fontsize = 15)
    ax.set_ylabel("Loss", fontsize = 15, rotation = 90)
    ax.set_yscale("log")
    ax.set_title('Loss Curve', fontsize = 15)

    plt.gcf().text(0.02, 0.9, title, fontsize=30)
