import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json

from sklearn.model_selection import GridSearchCV

from nn_lib import (
    Preprocessor,
    save_network,
    load_network,
)

from keras_model import create_model, r2_score


def evaluate_architecture(model, x_val, y_val):
    loss, metric = model.evaluate(x_val, y_val, batch_size=10)
    print("Mean squared error regression loss: {}".format(loss))
    print("R2 Score: {}".format(metric))
    return metric

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


def parameter_search(x_train_pre, y_train):
    model = KerasRegressor(build_fn=create_model, verbose=1)
    # define the grid search parameters
    batch_size = [100, 50, 10]
    epochs = [300, 200, 100]
    neurons = [500, 300, 100]
    activation = ["sigmoid", "relu"]
    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      neurons=neurons,
                      activation=activation)

    # Search for best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="r2", n_jobs=-1)
    grid_result = grid.fit(x_train_pre, y_train)

    params = grid_result.best_params_
    return params


def save_model(model):
    # Save model to json file
    with open("learn_FM.json", "w") as file:
        file.write(model.to_json())
    model.save_weights("learn_FM.h5")


def load_model():
    json_file = open("learn_FM.json", "r")
    model = json_file.read()
    json_file.close()

    model = model_from_json(model)
    model.load_weights("learn_FM.h5")
    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss='mean_squared_error',
            optimizer=sgd,
            metrics=[r2_score])

    return model


def main():
    # load data
    x_train_pre, y_train, x_val_pre, y_val = load_data("FM_dataset.dat")

    # Search for best hyperparameters
    params = parameter_search(x_train_pre, y_train)
    print("Best parameters: {p}".format(p=params))
    #params = {"neurons": 500, "activation": "sigmoid", "epochs": 300, "batch_size": 50}

    # Create a model using best params
    model = create_model(neurons=params['neurons'], activation=params['activation'])
    model.fit(x_train_pre, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    # Perform final evaluation on model
    evaluate_architecture(model, x_val_pre, y_val)

    # Save model
    save_model(model)

def predict_hidden(dat):
    # Preprocess data
    x = dat[:, :3]
    y = dat[:, 3:]
    prep_input = Preprocessor(dat)
    x_pre = prep_input.apply(x)

    model = load_model()
    pred = model.predict(x_pre)
    return pred


if __name__ == "__main__":
    dat = np.loadtxt("FM_dataset.dat")
    predict_hidden(dat)
