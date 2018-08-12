from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class AccLossPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph. 
    The graph is updated by the 'on_epoch_end' event of the Keras Callback class
    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event 
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0

    # def on_train_begin(self, logs={}):

    def on_epoch_end(self, epochs, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))



    def on_train_end(self, logs={}):

        epochs = [x for x in range(self.epoch_count)]

        plt.figure()
        plt.title('Accuracy')
        plt.plot(epochs, self.val_acc, color='r')
        plt.plot(epochs, self.acc, color='b')
        plt.ylabel('accuracy')
        red_patch = mpatches.Patch(color='red', label='Validation')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch], loc=4)
        plt.savefig('./plots/' + self.model_name + '/Accuracy.png')
        plt.close()

        plt.figure()
        plt.title('Loss')
        plt.plot(epochs, self.val_loss, color='r')
        plt.plot(epochs, self.loss, color='b')
        plt.ylabel('loss')
        red_patch = mpatches.Patch(color='red', label='Validation')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch], loc=4)
        plt.savefig('./plots/' + self.model_name + '/Loss.png')
        plt.close()