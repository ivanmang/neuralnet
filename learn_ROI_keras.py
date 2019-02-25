import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from nn_lib import (
    Preprocessor,
    save_network,
    load_network,
)

def load_data(filepath):
    # load data
    dat = np.loadtxt(filepath)

    # Shuffle data
    np.random.shuffle(dat)
    x = dat[:, :3]
    y = dat[:, 3:]

    # Split data into training and validation set
    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Preprocess data
    prep_input = Preprocessor(dat)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    return x_train_pre, y_train, x_val_pre, y_val

def main():

    # load data
    x_train_pre, y_train, x_val_pre, y_val = load_data("ROI_dataset.dat")

    model = Sequential()
    model.add(Dense(units=300, activation="relu", input_dim=3))
    model.add(Dense(units=4, activation="softmax"))

    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=["accuracy"])

    model.fit(x_train_pre, y_train, epochs=100, batch_size=50, verbose=1)

    loss, metric = model.evaluate(x_val_pre, y_val, batch_size=10)

    print(loss, metric)
    pred = model.predict(np.array([x_val_pre[0]]))
    print(pred, y_val[0])



if __name__ == "__main__":
    main()
