import pyDOE, sys, time, tqdm
import tensorflow_probability as tfp
from tensorflow.keras.layers import InputLayer, Dense, Layer
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def init_color_object():
    color = type('', (), {})() # initialize an empty object to contain colors in a terminal

    color.RESET = '\033[0m'
    color.BOLD = '\033[1m'
    color.DIM = '\033[2m'
    color.ITALIC = '\033[3m'
    color.UNDERLINE = '\033[4m'
    color.BLINK = '\033[5m'
    color.REVERSE = '\033[7m'
    color.HIDDEN = '\033[8m'
    
    color.BLUE   = "\033[94m"
    color.GREEN  = "\033[92m"
    color.YELLOW = "\033[93m"
    color.RESET  = "\033[0m"
    color.PURPLE = "\033[95m"
    color.RED    = "\033[91m"
    color.CYAN    = "\033[96m"
    color.ORANGE  = "\033[38;5;214m"
    color.GRAY    = "\033[90m"
    color.WHITE   = "\033[97m"
    color.LIGHT_GREEN = "\033[92m"
    color.LIGHT_BLUE  = "\033[94m"
    color.DARK_BLUE   = "\033[34m"
    color.LIGHT_RED   = "\033[91m"
    color.MAGENTA = "\033[95m"
    color.LIGHT_YELLOW = "\033[93m"
    color.BROWN   = "\033[38;5;94m"
    color.LIGHT_GRAY  = "\033[37m"

    color.BG_BLACK = '\033[40m'
    color.BG_RED = '\033[41m'
    color.BG_GREEN = '\033[42m'
    color.BG_YELLOW = '\033[43m'
    color.BG_BLUE = '\033[44m'
    color.BG_MAGENTA = '\033[45m'
    color.BG_CYAN = '\033[46m'
    color.BG_WHITE = '\033[47m'
    color.BG_BRIGHT_BLACK = '\033[100m'
    color.BG_BRIGHT_RED = '\033[101m'
    color.BG_BRIGHT_GREEN = '\033[102m'
    color.BG_BRIGHT_YELLOW = '\033[103m'
    color.BG_BRIGHT_BLUE = '\033[104m'
    color.BG_BRIGHT_MAGENTA = '\033[105m'
    color.BG_BRIGHT_CYAN = '\033[106m'
    color.BG_BRIGHT_WHITE = '\033[107m'

    return color


def plot(t_all, x_all, pred_all, t_train, x_train, model, title: str, vars=0.01, output_dir='.'):
    fig = plt.figure(figsize=(7, 12))

    ax = plt.subplot(311)
    ax.plot(t_all, x_all, "r", label="Exact Solution")
    ax.plot(t_all, pred_all, "b--", label="Prediction")
    ax.scatter(t_train, x_train, s=30, c="g", label="Training Data Point")
    ax.legend(fontsize=12)
    ax.set_xlabel("$t$", fontsize = 15)
    ax.set_ylabel("$x$", fontsize = 15, rotation = 0)
    ax.set_title(title, fontsize = 15)

    ax = plt.subplot(312)
    ax.plot(t_all, pred_all - x_all, "b-")
    ax.set_xlabel("$t$", fontsize = 15)
    ax.set_ylabel("Prediction - Exact Solution", fontsize = 15, rotation = 90)
    ax.set_title("Difference between Prediction and Exact Solution", fontsize = 15)

    ax = plt.subplot(313)
    loss_train = np.array(model.loss_history["train"])
    loss_test = np.array(model.loss_history["test"])
    ax.scatter(np.arange(loss_train.size) * 100, loss_train, s=75, marker="x", label="Train Loss")
    ax.scatter(np.arange(loss_train.size) * 100, loss_test, s=75, marker="+", label="Test Loss")
    ax.legend(fontsize=12)
    ax.set_xlabel("$iterations$", fontsize = 15)
    ax.set_ylabel("Loss", fontsize = 15, rotation = 90)
    ax.set_yscale("log")
    ax.set_title('Loss Curve', fontsize = 15)

    # plt.gcf().text(0.02, 0.9, title, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/curve_fitting_{vars}.png", dpi=300)