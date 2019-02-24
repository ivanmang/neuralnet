import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

from nn_lib import (
    Preprocessor,
    save_network,
    load_network,
)

from keras_model import create_model


def evaluate_architecture(model, x_val, y_val):
    loss, metric = model.evaluate(x_val, y_val, batch_size=10)
    print("Mean squared error regression loss: {}".format(loss))
    print("R2 Score: {}".format(metric))

    return metric

def main():
    dat = np.loadtxt("FM_dataset.dat")
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

    model = KerasRegressor(build_fn=create_model, verbose=1)

    # define the grid search parameters
    batch_size = [5, 10, 50]
    epochs = [50, 100, 200]
    neurons = [100, 200, 300]
    activation = ["relu", "sigmoid", "linear"]
    param_grid = dict(batch_size=batch_size,
                      epochs=epochs,
                      neurons=neurons,
                      activation=activation)

    # Search for best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="r2", n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)

    params = grid_result.best_params_
    print("Best parameters: {p}".format(p=params))

    # Best Parameters obtained:
    # - activation: sigmoid
    # - batch_size: 50
    # - epochs: 200
    # - neurons: 300

    # params = {"neurons": 300, "activation": "sigmoid", "epochs": 200, "batch_size": 50}

    # Create a model using best params
    model = create_model(neurons=params['neurons'], activation=params['activation'])
    model.fit(x_train_pre, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)


    metric = evaluate_architecture(model, x_val_pre, y_val)

if __name__ == "__main__":
    main()
