import json
import argparse

from tensorflow.python.keras import backend as K
from helper import *


def main():
    dir_setup()
    args = parse_command_line_args()

    if args.configpath:
        with open(args.configpath) as json_data_file:
            config = json.load(json_data_file)
        module_model = config['model']
        epochs = config['epochs']
        batch_size = config['batch_size']
        model_name = config['model'].split('.')[-1]

        model, xtrain, ytrain, validation_data = load_model(module_model, config)
        model.summary()
        train(model_name, model, xtrain, ytrain, validation_data, batch_size, epochs)
        prediction = predict(model, validation_data)
        plot_cm(config, validation_data, prediction)
        K.clear_session()

    else:
        configs = os.listdir("configs")
        for cfg in configs:
            if cfg[0] == '.':
                pass
            else:
                configpath = "./configs/" + cfg
                with open(configpath) as json_data_file:
                    config = json.load(json_data_file)
                module_model = config['model']
                epochs = config['epochs']
                batch_size = config['batch_size']
                model_name = config['model'].split('.')[-1]

                model, xtrain, ytrain, validation_data = load_model(module_model, config)
                model.summary()
                train(model_name, model, xtrain, ytrain, validation_data, batch_size, epochs)
                prediction = predict(model, validation_data)
                plot_cm(config, validation_data, prediction)
                K.clear_session()


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configpath',
                        help='The json path containing the target experiment configuration.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    print("The Mist is full of Mischief")


