import logging
from math import ceil, log2
import os

import numpy as np
from numpy import linalg
import scipy
from matplotlib import pyplot as plt
from numpy import diag, pi as PI
from paddle import fluid
from paddle import kron, matmul, trace
from paddle_quantum.circuit import UAnsatz, to_tensor
from paddle_quantum.utils import (dagger, partial_trace, pauli_str_to_matrix,
                                  state_fidelity)
from sklearn.decomposition import PCA
from tqdm import tqdm

# load the dataset
npz_file = np.load('datasets/pneumoniamnist.npz')
train_images = npz_file['train_images']
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = npz_file['train_labels']

test_images = npz_file['test_images']
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels = npz_file['test_labels']

# use PCA to reduce Dimension
pca = PCA(n_components=8)
new_train = pca.fit_transform(train_images)
new_test = pca.fit_transform(test_images)

# normalize
for i in range(len(new_train)):
    new_train[i] = new_train[i] / new_train[i].sum()

for i in range(len(new_test)):
    new_test[i] = new_test[i] / new_test[i].sum()
#                    ================================================QAE===============================================
# set up the circuit
N_A = 2  # the number of qubits in subsystem A
N_B = 1  # the number of qubits in subsystem B
N = N_A + N_B  # the total number of system

N=ceil(log2(train_images.shape[1]))
SEED = 14

scipy.random.seed(1)  # use fixed random seed
V = scipy.stats.unitary_group.rvs(2 ** N)  # randomly generate a unitary matrix V
V_H = V.conj().T  # V_dagger

cir_depth = 6  # the depth of the circuit
block_len = 2  # the length of each module
theta_size = N * block_len * cir_depth  # the shape of the parameter of the network

rho_C = np.diag([1, 0]).astype('complex128')


# Encoder


def Encoder(theta):
    # use UAnsatz to initialize the network
    cir = UAnsatz(N)

    for layer_num in range(cir_depth):

        for which_qubit in range(N):
            cir.ry(theta[block_len * layer_num * N + which_qubit], which_qubit)
            cir.rz(theta[(block_len * layer_num + 1)
                         * N + which_qubit], which_qubit)

        for which_qubit in range(N - 1):
            cir.cnot([which_qubit, which_qubit + 1])
        cir.cnot([N - 1, 0])

    return cir.U


def normalize2unitary(x):
    rho_in_mols = x
    rho_in_mols = x/linalg.norm(x)
    return rho_in_mols


def top_k_sum(arr, k):
    top_k_idx = arr.argsort()[::-1][0:k]
    top_k_sum = 0
    for idx in top_k_idx:
        top_k_sum += arr[idx]
    return top_k_sum


def plot_curve(loss, fid):
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(loss, label='train loss', marker="s")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('AE_train_ls1.png')
    plt.close()
    plt.xlabel("epochs")
    plt.ylabel("fid")
    plt.plot(fid, label='train fid', marker="s")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('AE_train_fid1.png')
    plt.close()


class QAE(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, param_attr=fluid.initializer.Uniform(
        low=0.0, high=2 * np.pi, seed=SEED), dtype='float64'):
        super(QAE, self).__init__()

        # Numpy array -> variable
        self.rho_C = fluid.dygraph.to_variable(rho_C)
        self.theta = self.create_parameter(shape=shape,
                                           attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, x):
        rho_in = fluid.dygraph.to_variable(x)
        E = Encoder(self.theta)
        E_dagger = dagger(E)
        D = E_dagger
        D_dagger = E

        rho_BA = matmul(matmul(E, rho_in), E_dagger)

        rho_encode = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 1)
        rho_trash = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 2)

        rho_CA = kron(self.rho_C, rho_encode)
        rho_out = matmul(matmul(D, rho_CA), D_dagger)

        zero_Hamiltonian = fluid.dygraph.to_variable(
            np.diag([1, 0]).astype('complex128'))
        loss = 1 - (trace(matmul(zero_Hamiltonian, rho_trash))).real()

        return loss, rho_out, rho_encode


#                    ================================================QClassifier===============================================


def myRy(theta):
    """
    :param theta: parameter
    :return: Y rotation matrix
    """
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])


def myRz(theta):
    """
    :param theta: parameter
    :return: Z rotation matrix
    """
    return np.array([[np.cos(theta / 2) - np.sin(theta / 2) * 1j, 0],
                     [0, np.cos(theta / 2) + np.sin(theta / 2) * 1j]])


def datapoints_transform_to_state(data, n_qubits):
    """
    :param data: shape [-1, 2]
    :param n_qubits: the number of qubits to which the data transformed
    :return: shape [-1, 1, 2 ^ n_qubits]
    """
    dim1 = data.shape
    res = []
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0]])
        for i in range(n_qubits):
            if i % 2 == 0:
                state_tmp = np.dot(zero_state, myRy(np.arcsin(data[sam][0])).T)
                state_tmp = np.dot(state_tmp, myRz(
                    np.arccos(data[sam][0] ** 2)).T)
                res_state = np.kron(res_state, state_tmp)
            elif i % 2 == 1:
                state_tmp = np.dot(zero_state, myRy(np.arcsin(data[sam][1])).T)
                state_tmp = np.dot(state_tmp, myRz(
                    np.arccos(data[sam][1] ** 2)).T)
                res_state = np.kron(res_state, state_tmp)
        res.append(res_state)

    res = np.array(res)
    return res.astype("complex128")


def U_theta(theta, n, depth):
    """
    :param theta: dim: [n, depth + 3]
    :param n: number of qubits
    :param depth: circuit depth
    :return: U_theta
    """
    cir = UAnsatz(n)

    for i in range(n):
        cir.rz(theta[i][0], i)
        cir.ry(theta[i][1], i)
        cir.rz(theta[i][2], i)

    for d in range(3, depth + 3):
        for i in range(n - 1):
            cir.cnot([i, i + 1])
        cir.cnot([n - 1, 0])
        for i in range(n):
            cir.ry(theta[i][d], i)

    return cir.U


def Observable(n):
    """
    :param n: number of qubits
    :return: local observable: Z \otimes I \otimes ...\otimes I
    """
    Ob = pauli_str_to_matrix([[1.0, 'z0']], n)
    return Ob


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 n,  # number of qubits
                 depth,  # circuit depth
                 seed_paras=1,
                 dtype='float64'):
        super(Net, self).__init__()

        self.n = n
        self.depth = depth

        self.theta = self.create_parameter(
            shape=[n, depth + 3],
            attr=fluid.initializer.Uniform(
                low=0.0, high=2 * PI, seed=seed_paras),
            dtype=dtype,
            is_bias=False)

        self.bias = self.create_parameter(
            shape=[1],
            attr=fluid.initializer.NormalInitializer(
                scale=0.01, seed=seed_paras + 10),
            dtype=dtype,
            is_bias=False)

    def forward(self, state_in, label):
        """
        Args:
            state_in: The input quantum state, shape [-1, 1, 2^n]
            label: label for the input state, shape [-1, 1]
        Returns:
            The loss:
                L = ((<Z> + 1)/2 + bias - label)^2
        """

        # Numpy array -> variable
        Ob = fluid.dygraph.to_variable(Observable(self.n))
        label_pp = fluid.dygraph.to_variable(label)

        Utheta = U_theta(self.theta, n=self.n, depth=self.depth)
        U_dagger = dagger(Utheta)

        state_out = matmul(matmul(state_in, Utheta), U_dagger)

        E_Z = matmul(state_out, Ob)

        # map <Z> to the predict label
        state_predict = E_Z.real() * 0.5 + 0.5 + self.bias
        loss = fluid.layers.reduce_mean((state_predict - label_pp) ** 2)

        is_correct = fluid.layers.where(
            fluid.layers.abs(state_predict - label_pp) < 0.5).shape[0]
        acc = is_correct / label.shape[0]

        return loss, acc, state_predict.numpy()


if __name__ == '__main__':
    log_name = 'pipeline.log'
    if os.path.exists(log_name):
        os.rename(dst=log_name + '.bak', src=log_name)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_name,
                        filemode='w')
    # pre train the QAE
    LR = 0.1  # 设置学习速率
    EPOCHS = 5
    # if not os.path.exists('autoencoder.pdparams'):
    #     logging.info('There is no pre-trained QAE, training QAE with LR={}, EPOCH={}'.format(LR, EPOCHS))
    #     with fluid.dygraph.guard():
    #         qae = QAE([theta_size])

    #         opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR,
    #                                                parameter_list=qae.parameters())

    #         tr_fid = []
    #         tr_ls = []
    #         best_fid = 0

    #         for epoch in range(EPOCHS):
    #             epoch_fid = []
    #             epoch_ls = []
    #             for i in tqdm(range(len((new_train)))):
    #                 x = new_train[i]
    #                 s = top_k_sum(x, 2 ** N_A)
    #                 trainx = normalize2unitary(x)
    #                 loss, rho_out, rho_encode = qae(trainx)

    #                 loss.backward()
    #                 opt.minimize(loss)
    #                 qae.clear_gradients()
    #                 fid = state_fidelity(trainx, rho_out.numpy()) / s
    #                 epoch_fid.append(fid)
    #                 epoch_ls.append(loss.numpy())
    #             tr_fid.append(np.square(np.array(epoch_fid).mean()))
    #             tr_ls.append(np.array(epoch_ls).mean())

    #             if best_fid < np.square(np.array(epoch_fid).mean()):
    #                 best_fid = np.square(np.array(epoch_fid).mean())
    #                 fluid.save_dygraph(qae.state_dict(), "autoencoder")

    #             msg = 'epoch: {}, loss: {:.4f}, fid: {:.4f}'.format(
    #                 str(epoch), np.array(epoch_ls).mean(), np.square(np.array(epoch_fid).mean()))
    #             print(msg)
    #             logging.info(msg)
    #         plot_curve(tr_ls, tr_fid)

    # logging.info('pre-trained QAE exists, traing Q-classifier')
    # train the classifier
    step = 1
    BATCH = 1
    EPOCH = 5
    total_loss = 0.0

    with fluid.dygraph.guard():
        net = Net(n=10, depth=3, seed_paras=19)
        opt = fluid.optimizer.AdamOptimizer(
            learning_rate=0.01, parameter_list=net.parameters())
        ae = QAE([theta_size])
        para_state_dict, _ = fluid.load_dygraph("autoencoder")
        tr_ls = []
        for epoch in range(EPOCH):
            epoch_ls = 0
            data_len = 0
            for i in tqdm(range(len(train_images))):
                step = step + 1
                ae.set_dict(para_state_dict)
                x = normalize2unitary(train_images[i])
                trainx = np.zeros(2**N)
                trainx[:x.shape[0]] = x

                # loss, rho_out, rho_encode = ae(trainx)
                rho_encode = to_tensor(trainx)
                inputy = (train_labels[i * BATCH:(i + 1) * BATCH].reshape(-1))
                trainy = np.asarray(inputy).astype('float64')
                loss, acc, state = net(state_in=rho_encode, label=trainy)

                total_loss += loss.numpy()[0]

                loss.backward()
                opt.minimize(loss)
                net.clear_gradients()
                epoch_ls += loss.numpy().sum()
                data_len += BATCH

                if (i + 1) % 200 == 0:
                    print(
                        '------------------------------TEST---------------------------------')
                    summary_test_correct = 0
                    for j in (range(len((new_test)))):
                        ae.set_dict(para_state_dict)
                        inputx = test_images[j]
                        x = normalize2unitary(inpux)
                        valx = np.zeros(2**N)
                        valx[:x.shape[0]]=x
                        # loss, rho_out, rho_encode = ae(trainx)
                        rho_encode = to_tensor(valx)
                        inputy = (
                            test_labels[j * BATCH:(j + 1) * BATCH].reshape(-1))
                        trainy = np.asarray(inputy).astype('float64')
                        loss, acc, state = net(
                            state_in=rho_encode, label=trainy)
                        is_correct = (np.abs(state.reshape(-1) - trainy) < 0.5) + 0

                        is_correct = is_correct.sum()

                        summary_test_correct = summary_test_correct + is_correct
                    msg1 = 'epoch: {}, [{}/{}]'.format(
                        str(epoch), summary_test_correct, len(test_labels))
                    print(msg1)
                    logging.info(msg1)
