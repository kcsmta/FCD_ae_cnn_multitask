import pickle
import os
import matplotlib.pyplot as plt

def plot_training_process(history, results_path, prob, data_part, net = "cnn"):
    # result_path = "./results"
    # pickle_file_name = 'FLOW016.pkl'
    #
    # history = pickle.load(open(pickle_file_name, 'rb'))
    try:
        train_loss = history['predict_layer_loss']
        train_acc = history['predict_layer_accuracy']
        val_loss = history['val_predict_layer_loss']
        val_acc = history['val_predict_layer_accuracy']
        train_loss_ae = history['decode_layer_loss']
        train_acc_ae = history['decode_layer_accuracy']
        val_loss_ae = history['val_decode_layer_loss']
        val_acc_ae = history['val_decode_layer_accuracy']
        Multitask = True
    except:
        Multitask = False


    if Multitask==False:
        train_loss = history['loss']
        train_acc = history['accuracy']
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']

        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(
            os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part * 100)) + '_accuracy.png'))
        plt.clf()
        # plt.show()

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(
            os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part * 100)) + '_loss.png'))
        plt.clf()
        # plt.show()

    # print(history)
    # print(train_loss)
    # print(type(train_loss))
    #
    # for epoch, val in enumerate(train_loss):
    #     print('epoch ' + str(epoch) + ", loss_value=" + str(val))

    else:
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part*100)) + '_predict_accuracy.png'))
        plt.clf()
        # plt.show()

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part*100)) + '_predict_loss.png'))
        plt.clf()
        # plt.show()

        plt.plot(train_acc_ae)
        plt.plot(val_acc_ae)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part * 100)) + '_ae_accuracy.png'))
        plt.clf()
        # plt.show()

        plt.plot(train_loss_ae)
        plt.plot(val_loss_ae)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(results_path, net + '_' + prob + "_" + str(int(data_part * 100)) + '_ae_loss.png'))
        plt.clf()
        # plt.show()
