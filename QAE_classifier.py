import pandas as pd
import numpy as np
import os
import pdb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm

import time
import matplotlib
import numpy as np
from numpy import pi as PI
from matplotlib import pyplot as plt

from paddle import fluid
from paddle.fluid.framework import ComplexVariable
from paddle.complex import matmul, transpose, kron
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import pauli_str_to_matrix

# Hyper_params
hyper_params = {
    'n_components': 3,
    'n_qubits': 3,
    'n_z': 2
}

npz_file = np.load('datasets/pneumoniamnist.npz')
train_data = npz_file['train_images']
train_labels = npz_file['train_labels']
val_data = npz_file['val_images']
val_labels = npz_file['val_labels']
test_data = npz_file['test_images']
test_labels = npz_file['test_labels']


def show_images(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(28, 28))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img, cmap='gray')
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


tsne = TSNE(n_components=hyper_params['n_components'], min_grad_norm=1e-5, init='pca', method='exact', angle=0.45,
            early_exaggeration=5, n_iter=1000)
pca = PCA(n_components=hyper_params['n_components'])

reduction_model = pca
all_code = np.concatenate([train_data, test_data], axis=0)
reduction_model.fit(all_code.reshape(all_code.shape[0], -1))
reduct_code = reduction_model.transform(all_code.reshape(all_code.shape[0], -1))

Q_code = reduct_code[:train_data.shape[0]]
Q1_code = reduct_code[train_data.shape[0]:]

train_len = Q_code.shape[0]
test_len = Q1_code.shape[0]

all_code = np.concatenate([Q_code, Q1_code], axis=0)
min_val = []
max_val = []

for i in range(hyper_params['n_components']):
    min_val.append(np.min(all_code[:, i]))
    max_val.append(np.max(all_code[:, i]))

pre_code = preprocessing.minmax_scale(all_code, feature_range=(-1.0 + 1e-5, 1.0 - 1e-5), axis=0)

QQQ_code = pre_code[:train_len]
QQQ1_code = pre_code[train_len:]


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


# 经典 -> 量子数据编码器
def datapoints_transform_to_state(data, n_qubits):
    """
    :param data: shape [-1, 2]
    :param n_qubits: the number of qubits to which the data transformed
    :return: shape [-1, 1, 2 ^ n_qubits]
    """
    dim1, dim2 = data.shape
    res = []
    origin_res = []
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0]])
        origin_state = []
        for k in range(n_qubits):
            state_tmp = np.dot(zero_state, myRy(np.arcsin(data[sam][k % hyper_params['n_components']])).T)
            state_tmp = np.dot(state_tmp, myRz(np.arccos(data[sam][k % hyper_params['n_components']] ** 2)).T)
            res_state = np.kron(res_state, state_tmp)
            origin_state.append(data[sam][k % hyper_params['n_components']])
        res.append(res_state)
        origin_res.append([np.array(origin_state)])

    res = np.array(res)
    origin_res = np.array(origin_res)
    return res.astype("complex128"), origin_res


# 经典 -> 量子数据编码器 (为分类器准备)
def datapoints_transform_to_state_for_classifier(data, n_qubits):
    """
    :param data: shape [-1, 2]
    :param n_qubits: the number of qubits to which the data transformed
    :return: shape [-1, 1, 2 ^ n_qubits]
    """
    dim1, dim2 = data.shape
    res = []
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0]])
        for i in range(n_qubits):
            state_tmp = np.dot(zero_state, myRy(np.arcsin(data[sam][i % hyper_params['n_components']])).T)
            state_tmp = np.dot(state_tmp, myRz(np.arccos(data[sam][i % hyper_params['n_components']] ** 2)).T)
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
    # 初始化网络
    cir = UAnsatz(n)

    # 先搭建广义的旋转层
    for i in range(n):
        cir.rz(theta[i][0], i)
        cir.ry(theta[i][1], i)
        cir.rz(theta[i][2], i)

    # 默认深度为 depth = 1
    # 搭建纠缠层和 Ry旋转层
    for d in range(3, depth + 3):
        for i in range(n - 1):
            cir.cnot([i, i + 1])
        cir.cnot([n - 1, 0])
        for i in range(n):
            cir.ry(theta[i][d], i)

    return cir.U


def AE_encoder(theta, n, z_n, depth):
    assert n >= z_n
    n_empty = n - z_n
    empty_half = fluid.dygraph.to_variable(np.eye(2 ** n_empty).astype('complex128'))
    encoder_half = U_theta(theta, n, depth)
    result = kron(empty_half, encoder_half)

    return result


def AE_decoder(theta, n, z_n, depth):
    assert n >= z_n
    n_empty = n - z_n
    empty_half = fluid.dygraph.to_variable(np.eye(2 ** n_empty).astype('complex128'))
    decoder_half = U_theta(theta, n, depth)
    result = kron(decoder_half, empty_half)

    return result


def Observable(n, measure_index=0):
    """
    :param n: number of qubits
    :return: local observable: Z \otimes I \otimes ...\otimes I
    """
    Ob = fluid.dygraph.to_variable(pauli_str_to_matrix([[1.0, 'z' + str(measure_index)]], n))
    return Ob


def transform_to_origin(a):
    for i in range(hyper_params['n_components']):
        a[i] = (a[i] + 1.0) / 2.0 * (max_val[i] - min_val[i]) + min_val[i]

    return a


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 n,  # number of qubits
                 n_z,
                 depth,  # circuit depth
                 seed_paras=1,
                 dtype='float64'):
        super(Net, self).__init__()

        self.n = n
        self.n_z = n_z
        self.depth = depth

        # 初始化参数列表 theta，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta1 = self.create_parameter(
            shape=[n, depth + 3],
            attr=fluid.initializer.Uniform(
                low=0.0, high=2 * PI, seed=seed_paras),
            dtype=dtype,
            is_bias=False)
        self.theta2 = self.create_parameter(
            shape=[n, depth + 3],
            attr=fluid.initializer.Uniform(
                low=0.0, high=2 * PI, seed=seed_paras),
            dtype=dtype,
            is_bias=False)

        # 初始化偏置 (bias)
        self.bias = self.create_parameter(
            shape=[1],
            attr=fluid.initializer.NormalInitializer(
                scale=0.01, seed=seed_paras + 10),
            dtype=dtype,
            is_bias=False)

        self.Ob = [Observable(2 * self.n - self.n_z, i) for i in range(self.n)]

    def encoder(self, state_in):
        # 按照随机初始化的参数 theta 
        Encoder = AE_encoder(self.theta1, n=self.n, z_n=self.n_z, depth=self.depth)

        # State in to input state
        initial_state = np.array([1] + [0] * (2 ** (self.n - self.n_z) - 1)).astype('complex128')
        initial_state = fluid.dygraph.to_variable(initial_state)
        input_state = kron(initial_state, state_in)

        # 因为 Utheta是学习得到的，我们这里用行向量运算来提速而不会影响训练效果
        state_z = matmul(input_state, Encoder)

        return state_z

    # 定义向前传播机制、计算损失函数 和交叉验证正确率
    def forward(self, state_in, origin_state):
        """
        Args:
            state_in: The input quantum state, shape [-1, 1, 2^n]
            label: label for the input state, shape [-1, 1]
        Returns:
            The loss:
                L = ((<Z> + 1)/2 + bias - label)^2
        """

        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        Ob = self.Ob

        # 按照随机初始化的参数 theta 
        Encoder = AE_encoder(self.theta1, n=self.n, z_n=self.n_z, depth=self.depth)
        Decoder = AE_decoder(self.theta2, n=self.n, z_n=self.n_z, depth=self.depth)

        # State in to input state
        initial_state = np.array([1] + [0] * (2 ** (self.n - self.n_z) - 1)).astype('complex128')
        initial_state = fluid.dygraph.to_variable(initial_state)
        input_state = kron(initial_state, state_in)

        # 因为 Utheta是学习得到的，我们这里用行向量运算来提速而不会影响训练效果
        state_z = matmul(input_state, Encoder)
        state_out = matmul(state_z, Decoder)

        # 测量得到泡利 Z 算符的期望值 <Z>
        E_Z = [matmul(matmul(state_out, Ob[i]),
                      transpose(ComplexVariable(state_out.real, -state_out.imag),
                                perm=[0, 2, 1])).real for i in range(self.n)]

        output_state = fluid.layers.concat(E_Z, axis=-1)

        # Calcualate Loss
        loss = fluid.layers.mean((output_state - origin_state) ** 2)

        origin_len = fluid.layers.reduce_sum(origin_state ** 2, -1) ** 0.5
        output_len = fluid.layers.reduce_sum(output_state ** 2, -1) ** 0.5
        dot_product = fluid.layers.reduce_sum(output_state * origin_state, -1)

        fidelity = fluid.layers.mean(dot_product / origin_len / output_len)

        return loss, fidelity, output_state.numpy()


class Classifier(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 n,  # number of qubits
                 n_z,
                 depth,  # circuit depth
                 net,
                 seed_paras=1,
                 dtype='float64'):
        super(Classifier, self).__init__()

        self.n = n
        self.n_z = n_z
        self.depth = depth
        self.net = net

        # 初始化参数列表 theta，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(
            shape=[self.n_z, depth + 3],
            attr=fluid.initializer.Uniform(
                low=0.0, high=2 * PI, seed=seed_paras),
            dtype=dtype,
            is_bias=False)

        # 初始化偏置 (bias)
        self.bias = self.create_parameter(
            shape=[1],
            attr=fluid.initializer.NormalInitializer(
                scale=0.01, seed=seed_paras + 10),
            dtype=dtype,
            is_bias=False)

    # 定义向前传播机制、计算损失函数 和交叉验证正确率
    def forward(self, state_in, label):
        """
        Args:
            state_in: The input quantum state, shape [-1, 1, 2^n]
            label: label for the input state, shape [-1, 1]
        Returns:
            The loss:
                L = ((<Z> + 1)/2 + bias - label)^2
        """

        state_z = self.net.encoder(state_in)
        state_z = fluid.dygraph.to_variable(state_z.numpy())

        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        unused_n = self.n - self.n_z
        Ob = fluid.dygraph.to_variable(Observable(2 * self.n - self.n_z, measure_index=unused_n))
        label_pp = fluid.dygraph.to_variable(label)
        # 按照随机初始化的参数 theta 
        unused_n = self.n - self.n_z
        Utheta = U_theta(self.theta, n=self.n_z, depth=self.depth)
        empty_half = fluid.dygraph.to_variable(np.eye(2 ** unused_n).astype('complex128'))
        Utheta = kron(empty_half, Utheta)
        Utheta = kron(Utheta, empty_half)

        # 因为 Utheta是学习得到的，我们这里用行向量运算来提速而不会影响训练效果
        state_out = matmul(state_z, Utheta)  # 维度 [-1, 1, 2 ** n]

        # 测量得到泡利 Z 算符的期望值 <Z>
        E_Z = matmul(matmul(state_out, Ob),
                     transpose(ComplexVariable(state_out.real, -state_out.imag),
                               perm=[0, 2, 1]))

        # 映射 <Z> 处理成标签的估计值 
        state_predict = E_Z.real[:, 0] * 0.5 + 0.5 + self.bias
        loss = fluid.layers.reduce_mean((state_predict - label_pp) ** 2)

        return loss, state_predict.numpy()


def train_autoencoder():
    print('Training Autoencoder...')
    step = 1
    BATCH = 10
    EPOCH = 1
    total_loss = 0.0
    total_fidelity = 0.0
    fidelity_cnt = 0

    with fluid.dygraph.guard():
        opt = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())

        tr_ls = []

        step_arr = []
        loss_arr = []

        for epoch in range(EPOCH):
            print('Epoch', epoch)
            epoch_ls = 0
            data_len = 0
            for i in range(train_len // BATCH):
                step = step + 1
                inputx = QQQ_code[i * BATCH:(i + 1) * BATCH]
                inputx = np.asarray(inputx)

                trainx = np.array(inputx).astype('float64')
                res, origin_res = datapoints_transform_to_state(trainx, n_qubits=hyper_params['n_qubits'])
                input_data = fluid.dygraph.to_variable(res)
                origin_res = fluid.dygraph.to_variable(origin_res)
                # print(input_data)
                inputy = (train_labels[i * BATCH:(i + 1) * BATCH].reshape(-1))
                # pdb.set_trace()
                trainy = np.asarray(inputy).astype('float64')
                # print('label:--',trainy)
                loss, fidelity, state = net(state_in=input_data, origin_state=origin_res)
                total_loss += loss.numpy()[0]
                total_fidelity += fidelity.numpy()[0]
                fidelity_cnt += 1

                if i % 20 == 0:
                    print(f'Epoch:{epoch}, Step:{i}, Loss:{loss.numpy()[0]}, Fidelity:{total_fidelity / fidelity_cnt}')

                    fidelity_cnt = total_fidelity = 0

                    # Show the pictures
                    pic = transform_to_origin(state[0, 0])
                    output_img = reduction_model.inverse_transform(pic).reshape(28, 28)
                    plt.imshow(output_img, cmap='gray')
                    plt.title(f'QAE Step: {step}')
                    plt.savefig('ae.png')
                    plt.close()

                    plt.imshow(train_data[i * BATCH], cmap='gray')
                    plt.title(f'Origin Step: {step}')
                    plt.savefig('origin.png')
                    plt.close()

                    step_arr.append(step)
                    loss_arr.append(loss.numpy()[0])
                    plt.plot(step_arr, loss_arr)
                    plt.title('Loss')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.savefig('loss.png')
                    plt.close()

                loss.backward()
                opt.minimize(loss)
                net.clear_gradients()
                epoch_ls += loss.numpy().sum()
                data_len += BATCH

            tr_ls.append(epoch_ls / data_len)
            print('Loss:', epoch_ls / data_len)
        print(tr_ls)

    return net


with fluid.dygraph.guard():
    net = Net(n=hyper_params['n_qubits'], depth=3, n_z=hyper_params['n_z'], seed_paras=19)
    print('Loading autoencoder...')
    try:
        state_dict, _ = fluid.load_dygraph(f'qae_net_{hyper_params["n_qubits"]}_{hyper_params["n_z"]}')
        net.set_dict(state_dict)
    except Exception as err:
        print(err)
        train_autoencoder()
        fluid.save_dygraph(net.state_dict(), f'qae_net_{hyper_params["n_qubits"]}_{hyper_params["n_z"]}')

    print('Finished. Training Classifier...')

    classifier = Classifier(n=hyper_params['n_qubits'], depth=3, n_z=hyper_params['n_z'], seed_paras=19, net=net)
    opt = fluid.optimizer.AdamOptimizer(learning_rate=0.05, parameter_list=classifier.parameters())

    step = 1
    BATCH = 2
    EPOCH = 1
    total_loss = 0.0

    tr_ls = []
    for epoch in range(EPOCH):
        print('Epoch', epoch)
        epoch_ls = 0
        data_len = 0
        loss_arr = []

        for i in range(train_len // BATCH):
            step = step + 1
            inputx = QQQ_code[i * BATCH:(i + 1) * BATCH]
            inputx = np.asarray(inputx)

            trainx = np.array(inputx).astype('float64')
            input_data = fluid.dygraph.to_variable(
                datapoints_transform_to_state_for_classifier(trainx, n_qubits=hyper_params['n_qubits']))
            # print(input_data)
            inputy = (train_labels[i * BATCH:(i + 1) * BATCH].reshape(-1))
            # pdb.set_trace()
            trainy = np.asarray(inputy).astype('float64')
            # print('label:--',trainy)
            loss, state = classifier(state_in=input_data, label=trainy)

            total_loss += loss.numpy()[0]

            loss.backward()
            opt.minimize(loss)
            classifier.clear_gradients()
            epoch_ls += loss.numpy().sum()
            data_len += BATCH

            if (i) % 200 == 0:
                print('------------------------------TEST---------------------------------')
                summary_test_correct = 0
                for j in tqdm(range(test_len // BATCH)):
                    # lll=test_data[j]
                    label_test = (test_labels[j * BATCH:(j + 1) * BATCH].reshape(-1))
                    label_test = np.asarray(label_test).astype('float64')
                    # input_test=[]
                    # input_test.append(QQQ1_code[j])
                    input_test = QQQ1_code[j * BATCH:(j + 1) * BATCH]
                    input_test = np.asarray(input_test)
                    input_test = np.array(input_test).astype('float64')
                    # pdb.set_trace()
                    input_test = fluid.dygraph.to_variable(
                        datapoints_transform_to_state_for_classifier(input_test, n_qubits=hyper_params['n_qubits']))
                    loss, state = classifier(state_in=input_test, label=label_test)
                    is_correct = (np.abs(state.reshape(-1) - label_test) < 0.5) + 0
                    is_correct = is_correct.sum()
                    # pdb.set_trace()
                    summary_test_correct = summary_test_correct + is_correct
                print(epoch, summary_test_correct, test_len)
                # print( epoch ,acc, test_acc)
        tr_ls.append(epoch_ls / data_len)
        print('Loss:', epoch_ls / data_len)
    print(tr_ls)
