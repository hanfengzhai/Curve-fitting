import sys, os
sys.path.append('src')
from ml_utils import NeuralNet, TrainNeuralNet
from visualization_utils import init_color_object

import matplotlib, tensorflow as tf # import just to print in consistent with the colab HW format
print("Below are the versions that this Colab notebook (for me it's python :)) uses:")
print("Python version: " + sys.version); import matplotlib
print("matplotlib version: " + matplotlib.__version__)
print("TensorFlow version: " + tf.__version__)

colors = init_color_object()
output_dir = "output_HW1"
os.makedirs(output_dir, exist_ok=True)

""" ### Tasks
1. Find the learning rate that makes the loss converge (DONE)
2. Find the minimum amount of training data needed to make the loss converge (DONE)
3. Effects of activation function
4. Effects of noisy data (e.g. Gausian noise)
5. Find the parameters that can cause overfitting (test loss increases over iterations). The maximum iterations you can use is 20k.
"""

if __name__ == "__main__":
    # default parameters at the beginning - we will now start to tune them to different tasks
    n_total, train_ratio = 1000, 0.8
    n_train = int(n_total * train_ratio)
    lr, epoch = 0.01, 2000

    # Task #1 - find optimal learning rate
    for _lr_ in [1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2]:
        train_nn = TrainNeuralNet(n_total=n_total, n_train=n_train, lr=_lr_, epoch=epoch, output_dir=output_dir)
        train_nn.prepare_data()
        train_nn.train_model()
        train_nn.post_process(option='lr')
        train_nn.print_info()
    print(f'{colors.GREEN}Finish tuning the learning rate{colors.RESET}')
    
    # Task #2 - find minimum data required
    for _train_ratio_ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
        _n_train_ = int(n_total * _train_ratio_)
        train_nn = TrainNeuralNet(n_total=n_total, n_train=_n_train_, lr=lr, epoch=epoch, output_dir=output_dir)
        train_nn.prepare_data()
        train_nn.train_model()
        train_nn.post_process(option='train_ratio')
        train_nn.print_info()
    print(f'{colors.GREEN}Finish tuning the data ratio{colors.RESET}')
    
    # Task #3 - play with activation function
    train_nn = TrainNeuralNet(n_total=n_total, n_train=n_train, lr=lr, epoch=epoch, 
                              actv_func='tanh', output_dir=output_dir)
    train_nn.prepare_data()
    train_nn.train_model()
    train_nn.post_process(option='activation')
    train_nn.print_info()
    print(f'{colors.GREEN}Finish tuning the activation function{colors.RESET}')