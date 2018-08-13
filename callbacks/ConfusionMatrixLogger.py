from keras.callbacks import Callback
from helper import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def make_movie(images, path, width, height):
    # video_name = path + '/evolutionary_confusion_matrix.m4v'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(path, fourcc, 3, (width, height))

    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """

    def __init__(self, X, Y, classes, model_name):
        super().__init__()

        self.X = X
        self.Y = Y
        self.classes = classes
        self.model_name = model_name
        self.path = './plots/' + self.model_name + '/evolutionary_confusion_matrix.m4v'
        self.path_norm = './plots/' + self.model_name + '/evolutionary_confusion_matrix_normalized.m4v'

        self.images = []
        self.images_norm = []
        self.width = 0
        self.height = 0


    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        predictions = self.model.predict(self.X, verbose=0)
        plot_cm(self.model_name, self.classes, self.Y, predictions)
        self.images.append(cv2.imread(self.path))
        self.images_norm.append(cv2.imread(self.path_norm))


    def on_train_end(self, logs={}):
        cm_path = './plots/' + self.model_name + '/confusion_matrix_non_normalized.png'
        self.height, self.width, _ = cv2.imread(cm_path).shape
        print("self.height",self.height)
        print("self.width",self.width)

        make_movie(self.images, self.path, self.width, self.height)
        make_movie(self.images_norm, self.path_norm, self.width, self.height)


