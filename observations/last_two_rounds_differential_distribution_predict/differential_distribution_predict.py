import os
import sys

import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264

cipher_dict = {
    "speck3264":speck3264
}


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
    return res


bs = 5000


def make_predict_net(model, num_outputs=12, ds=[64, 64], reg_param=10**-5, final_activation='sigmoid'):
    # 冻结 model 的所有层
    for layer in model.layers:
        layer.trainable = False
    model = Model(inputs=model.input, outputs=model.layers[-8].output)
    dense = model.output
    for i, d in enumerate(ds):
        dense = Dense(d,kernel_regularizer=l2(reg_param), name=f'predict_net_dense_{i}')(dense)
        dense = BatchNormalization(name=f'predict_net_batch_norm_{i}')(dense)
        dense = Activation('relu', name=f'predict_net_activation_{i}')(dense)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense)
    return Model(inputs=model.input, outputs=out)


def train_predict_distinguisher(index, cipher, num_outputs, num_rounds=5, backwards_num=2, num_epochs=20, num_dataset=10**7, diff=(0x0040,0), ds=[64, 64], reg_param=10**-5, loss='mse', wdir="./freshly_trained_nets/"):
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"], 
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        model = load_model('./best5depth10.h5')
        net = make_predict_net(model, num_outputs, ds, reg_param)
        net.compile(optimizer='adam', loss=loss, metrics=['binary_accuracy'])
        # net.summary()
    X, Y = cipher.make_train_data_for_predict(num_dataset,num_rounds,backwards_num,diff)
    X_eval, Y_eval = cipher.make_train_data_for_predict(num_dataset//10,num_rounds,backwards_num,diff)
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'backwards'+str(backwards_num)+'_'+str(index)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    print("Best validation accuracy: ", np.max(h.history['val_binary_accuracy']))
    return net, h


def eval_for_diff(net, cipher, eval_dataset_size, num_rounds, backwards_num, diff):
    X, Y = cipher.make_train_data_for_predict(eval_dataset_size, num_rounds, backwards_num, diff);

    Y_pred = net.predict(X)
    Y_pred_bool = (Y_pred > 0.5).astype(int)
    accuracy = np.mean(np.all(Y_pred_bool == Y, axis=1))
    
    bit_diff = np.sum(Y_pred_bool != Y, axis=1)
    diff_counts = {i: np.sum(bit_diff == i) for i in range(0, 13)}
    
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    for i in range(0, 13):
        print(f"Number of samples with {i} bit(s) difference: {diff_counts[i]}")


def extract_truncate_differential(Y):
    truncate_to_extract = [0, 1, 7, 8, 14, 15, 16, 17, 23, 24, 30, 31]
    return Y[:, truncate_to_extract]


def eval_for_truncate_diff(net, cipher, eval_dataset_size, num_rounds, backwards_num, diff):
    X, Y = cipher.make_train_data_for_predict(eval_dataset_size, num_rounds, backwards_num, diff)
    Y = extract_truncate_differential(Y)

    Y_pred = net.predict(X)
    Y_pred_bool = (Y_pred > 0.5).astype(int)
    Y_pred_bool = extract_truncate_differential(Y_pred_bool)
    accuracy = np.mean(np.all(Y_pred_bool == Y, axis=1))
    
    bit_diff = np.sum(Y_pred_bool != Y, axis=1)
    diff_counts = {i: np.sum(bit_diff == i) for i in range(0, 13)}
    
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    for i in range(0, 13):
        print(f"Number of samples with {i} bit(s) difference: {diff_counts[i]}")


if __name__ == '__main__':
    for index in range(10):
        net, h = train_predict_distinguisher(index=0, cipher=cipher_dict['speck3264'], num_outputs=32, num_rounds=5, backwards_num=2, num_epochs=40, num_dataset=10**7, diff=(0x0040,0), ds=[64,64], reg_param=10**-5, loss='binary_crossentropy')
        print('eval_for_diff: ')
        eval_for_diff(net=net, cipher=cipher_dict['speck3264'], eval_dataset_size=10**6, num_rounds=5, backwards_num=2, diff=(0x0040,0))
        print('eval_for_truncate_diff: ')
        eval_for_truncate_diff(net=net, cipher=cipher_dict['speck3264'], eval_dataset_size=10**6, num_rounds=5, backwards_num=2, diff=(0x0040,0))
        