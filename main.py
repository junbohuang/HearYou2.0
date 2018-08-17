import json
import argparse

from tensorflow.python.keras import backend as K
from helper import *
from plotters.confusion_matrix import *


def main():
    dir_setup()
    args = parse_command_line_args()

    if args.configpath:
        with open(args.configpath) as json_data_file:
            config = json.load(json_data_file)

        model, xtrain, ytrain, xtest, ytest = load_model(config)

        model.summary()
        train(config, model, xtrain, ytrain, xtest, ytest)
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

                model, xtrain, ytrain, xtest, ytest = load_model(config)
                model.summary()
                train(config, model, xtrain, ytrain, xtest, ytest)
                # evaluate(config, model, xtest, ytest)
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


