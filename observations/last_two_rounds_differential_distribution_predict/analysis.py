import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import os
import sys

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264

cipher_dict = {
    "speck3264":speck3264
}


def diff_bias_analysis(cipher, eval_dataset_size, num_rounds, backwards_num, diff):
    X, Y = cipher_dict['speck3264'].make_train_data_for_predict(eval_dataset_size, num_rounds, backwards_num, diff)
    return np.mean(Y, axis=0)


def analysis(net, cipher, eval_dataset_size, num_rounds, backwards_num, diff):
    X, Y = cipher.make_train_data_for_predict(eval_dataset_size, num_rounds, backwards_num, diff)

    Y_pred = net.predict(X, batch_size=10000)
    Y_pred_bool = (Y_pred > 0.5).astype(int)

    accuracy_per_dim = []
    for i in range(Y.shape[1]):
        correct_predictions = np.sum(Y_pred_bool[:, i] == Y[:, i])
        accuracy = correct_predictions / eval_dataset_size
        accuracy_per_dim.append(accuracy)

    return accuracy_per_dim


def show_combined_analysis(acc_input, bias_input):
    # Bit positions for which we will plot bias with points
    special_bits = {0, 1, 7, 8, 14, 15, 16, 17, 23, 24, 30, 31}

    fig, ax1 = plt.subplots(figsize=(10, 6)) 

    # Plot accuracy on the left y-axis, with larger markers and line width
    ax1.plot(range(len(acc_input)), acc_input, marker='o', linestyle='-', color='orange', markersize=8, linewidth=2) 
    ax1.set_xlabel('Bit')
    ax1.set_ylabel('Accuracy', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('Accuracy and Bias per Bit') 
    ax1.grid(True)

    # Change the red horizontal line at acc = 0.95 to a solid line, matching the width of the accuracy line
    ax1.plot([0, len(acc_input)-1], [0.95, 0.95], color='red', linestyle='-', linewidth=2)

    # Create a second y-axis to plot bias
    ax2 = ax1.twinx()

    # Plot bias as a dashed line
    ax2.plot(range(len(bias_input)), bias_input, linestyle='--', color='deepskyblue', alpha=0.6)

    # Plot bias with '*' for most positions, but use 'o' (points) for special bits
    for i in range(len(bias_input)):
        if i in special_bits:
            ax2.plot(i, bias_input[i], marker='o', color='deepskyblue', linestyle='')  # 'o' for special bits
        else:
            ax2.plot(i, bias_input[i], marker='*', color='deepskyblue', linestyle='')  # '*' for other bits

    ax2.set_ylabel('Bias', color='deepskyblue')
    ax2.tick_params(axis='y', labelcolor='deepskyblue')

    # Set x-ticks to match the number of bits
    plt.xticks(np.arange(len(acc_input)))

    # Adjust the x-axis limits to ensure the red line spans the entire width of the plot
    ax1.set_xlim(-0.5, len(acc_input) - 0.5)

    # Save the combined plot without legend
    # plt.savefig('analysis.png')
    # plt.close()
    # print(f"The combined plot  has been saved at analysis.png")
    plt.savefig('analysis.pdf')
    print(f"The combined plot  has been saved at analysis.pdf")


if __name__ == '__main__':
    net0 = load_model('./freshly_trained_nets/best5backwards2_0.h5')
    accuracy_per_dim_0 = analysis(net0, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net0 finished")
    net1 = load_model('./freshly_trained_nets/best5backwards2_1.h5')
    accuracy_per_dim_1 = analysis(net1, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net1 finished")
    net2 = load_model('./freshly_trained_nets/best5backwards2_2.h5')
    accuracy_per_dim_2 = analysis(net2, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net2 finished")
    net3 = load_model('./freshly_trained_nets/best5backwards2_3.h5')
    accuracy_per_dim_3 = analysis(net3, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net3 finished")
    net4 = load_model('./freshly_trained_nets/best5backwards2_4.h5')
    accuracy_per_dim_4 = analysis(net4, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net4 finished")
    net5 = load_model('./freshly_trained_nets/best5backwards2_5.h5')
    accuracy_per_dim_5 = analysis(net5, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net5 finished")
    net6 = load_model('./freshly_trained_nets/best5backwards2_6.h5')
    accuracy_per_dim_6 = analysis(net6, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net6 finished")
    net7 = load_model('./freshly_trained_nets/best5backwards2_7.h5')
    accuracy_per_dim_7 = analysis(net7, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net7 finished")
    net8 = load_model('./freshly_trained_nets/best5backwards2_8.h5')
    accuracy_per_dim_8 = analysis(net8, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net8 finished")
    net9 = load_model('./freshly_trained_nets/best5backwards2_9.h5')
    accuracy_per_dim_9 = analysis(net9, cipher_dict['speck3264'], 10**6, 5, 2, (0x0040, 0))
    print("net9 finished")
    accuracy_per_dim = [accuracy_per_dim_0, accuracy_per_dim_1, accuracy_per_dim_2, accuracy_per_dim_3, accuracy_per_dim_4, accuracy_per_dim_5, accuracy_per_dim_6, accuracy_per_dim_7, accuracy_per_dim_8, accuracy_per_dim_9]
    print("accuracy_per_dim: \n", accuracy_per_dim)
    avg_accuracy_per_dim = np.mean(np.vstack(accuracy_per_dim), axis=0)
    print("avg_accuracy_per_dim: \n", avg_accuracy_per_dim)
    
    bias_per_dim = diff_bias_analysis(cipher_dict['speck3264'], 10**7, 5, 2, (0x0040, 0))
    
    show_combined_analysis(avg_accuracy_per_dim, bias_per_dim - 0.5)
