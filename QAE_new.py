import os

import numpy as np
import paddle
import pandas as pd
import scipy
import copy
import pdb

from collections import OrderedDict
from matplotlib import pyplot as plt
from numpy import diag
from paddle import fluid
from paddle.complex import kron, matmul, trace
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger, partial_trace, state_fidelity
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm



import time
import matplotlib
import numpy as np
from numpy import pi as PI
from matplotlib import pyplot as plt

from paddle import fluid
from paddle.fluid.framework import ComplexVariable
from paddle.complex import matmul, transpose
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import pauli_str_to_matrix

npz_file = np.load('datasets/minidata.npz')
train_images = npz_file['train_images']
train_images = train_images.reshape(train_images.shape[0], -1)

val_images = npz_file['val_images']
val_images = val_images.reshape(val_images.shape[0], -1)

pca = PCA(n_components=8)
new_train = pca.fit_transform(train_images)
new_val = pca.fit_transform(val_images)



# 保证每张图片的向量所有元素之和为1
for i in range(len(new_train)):
    new_train[i] = new_train[i] / new_train[i].sum()
    
for i in range(len(new_val)):
    new_val[i] = new_val[i] / new_val[i].sum()


N_A = 2        # 系统 A 的量子比特数
N_B = 1        # 系统 B 的量子比特数
N = N_A + N_B  # 总的量子比特数

scipy.random.seed(1)                            # 固定随机种子
V = scipy.stats.unitary_group.rvs(2**N)         # 随机生成一个酉矩阵
V_H = V.conj().T                                # 进行厄尔米特转置



# 设置电路参数
cir_depth = 6                        # 电路深度
block_len = 2                        # 每个模组的长度
theta_size = N*block_len*cir_depth   # 网络参数 theta 的大小


# 搭建编码器 Encoder E
def Encoder(theta):

    # 用 UAnsatz 初始化网络
    cir = UAnsatz(N)
    
    # 搭建层级结构：
    for layer_num in range(cir_depth):
        
        for which_qubit in range(N):
            cir.ry(theta[block_len*layer_num*N + which_qubit], which_qubit)
            cir.rz(theta[(block_len*layer_num + 1)*N + which_qubit], which_qubit)

        for which_qubit in range(N-1):
            cir.cnot([which_qubit, which_qubit + 1])
        cir.cnot([N-1, 0])

    return cir.U



def normalize2unitary(x):
    rho_in_mols=x
    rho_in_mols=(V@diag(rho_in_mols)@V_H).astype('complex128')
    return rho_in_mols



def top_k_sum(arr, k):
    top_k_idx = arr.argsort()[::-1][0:k]
    top_k_sum = 0
    for idx in top_k_idx:
        top_k_sum += arr[idx]
    return top_k_sum



rho_C = np.diag([1,0]).astype('complex128')



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
    

N_A = 2        # 系统 A 的量子比特数
N_B = 1        # 系统 B 的量子比特数
N = N_A + N_B  # 总的量子比特数
# ITR = 100      # 设置迭代次数
SEED = 14      # 固定初始化参数用的随机数种子

class NET4(fluid.dygraph.Layer):
    """
    Construct the model net
    """
    def __init__(self, shape, param_attr=fluid.initializer.Uniform(
        low=0.0, high=2 * np.pi, seed = SEED), dtype='float64'):
        super(NET4, self).__init__()
        
        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        self.rho_C = fluid.dygraph.to_variable(rho_C)
        self.theta = self.create_parameter(shape=shape, 
                     attr=param_attr, dtype=dtype, is_bias=False)
    
    # 定义损失函数和前向传播机制
    def forward(self,x):
        # 生成初始的编码器 E 和解码器 D\n",
        rho_in= fluid.dygraph.to_variable(x)
        E = Encoder(self.theta)
        E_dagger = dagger(E)
        D = E_dagger
        D_dagger = E

        # 编码量子态 rho_in
        rho_BA = matmul(matmul(E, rho_in), E_dagger)
        
        # 取 partial_trace() 获得 rho_encode 与 rho_trash
        rho_encode = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 1)
        rho_trash = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 2)

        # 解码得到量子态 rho_out
        rho_CA = kron(self.rho_C, rho_encode)
        rho_out = matmul(matmul(D, rho_CA), D_dagger)
        
        # 通过 rho_trash 计算损失函数
        
        zero_Hamiltonian = fluid.dygraph.to_variable(np.diag([1,0]).astype('complex128'))
        loss = 1 - (trace(matmul(zero_Hamiltonian, rho_trash))).real

        return loss, rho_out, rho_encode



LR = 0.01  # 设置学习速率
EPOCHS = 1

print('Start training...')
with fluid.dygraph.guard():
    net = NET4([theta_size])

    opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR,
                          parameter_list=net.parameters())

    tr_fid = []
    tr_ls = []
    best_fid = 0
    
    for epoch in range(EPOCHS):
        epoch_fid = []
        epoch_ls = []
        for i in tqdm(range(len((new_train)))):
            x=new_train[i]
            s=top_k_sum(x, 2**N_A)
            trainx=normalize2unitary(x)
            loss, rho_out, rho_encode=net(trainx)

            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()
            fid=state_fidelity(trainx, rho_out.numpy()) / s
            epoch_fid.append(fid)
            epoch_ls.append(loss.numpy())
        tr_fid.append(np.square(np.array(epoch_fid).mean()))
        tr_ls.append(np.array(epoch_ls).mean())
        
        if best_fid < np.square(np.array(epoch_fid).mean()):
            best_fid=np.square(np.array(epoch_fid).mean())
            fluid.save_dygraph(net.state_dict(), "autoencoder")

        print('epoch:', epoch, 'loss:', '%.4f' % np.array(epoch_ls).mean(),
              'fid:', '%.4f' % np.square(np.array(epoch_fid).mean()))
    plot_curve(tr_ls, tr_fid)


with fluid.dygraph.guard():
    ae = NET4([theta_size])
    para_state_dict, _ = fluid.load_dygraph("autoencoder")
    ae.set_dict(para_state_dict)
    x=new_train[1]
    s=top_k_sum(x, 2**N_A)
    trainx=normalize2unitary(x)
    loss, rho_out, rho_encode = ae(trainx)


rho_encode.numpy()


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
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0]])
        for i in range(n_qubits):
            if i % 2 == 0:
                state_tmp=np.dot(zero_state, myRy(np.arcsin(data[sam][0])).T)
                state_tmp=np.dot(state_tmp, myRz(np.arccos(data[sam][0] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
            elif i % 2 == 1:
                state_tmp=np.dot(zero_state, myRy(np.arcsin(data[sam][1])).T)
                state_tmp=np.dot(state_tmp, myRz(np.arccos(data[sam][1] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
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
        for i in range(n-1):
            cir.cnot([i, i + 1])
        cir.cnot([n-1, 0])
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
                 n,      # number of qubits
                 depth,  # circuit depth
                 seed_paras=1,
                 dtype='float64'):
        super(Net, self).__init__()

        self.n = n
        self.depth = depth
        
        # 初始化参数列表 theta，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(
            shape=[n, depth + 3],
            attr=fluid.initializer.Uniform(
                low=0.0, high=2*PI, seed=seed_paras),
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
        #pdb.set_trace()
        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        Ob = fluid.dygraph.to_variable(Observable(self.n))
        
        label_pp = fluid.dygraph.to_variable(label)
        # 按照随机初始化的参数 theta 
        Utheta = U_theta(self.theta, n=self.n, depth=self.depth)
        U_dagger = dagger(Utheta)
        # 因为 Utheta是学习得到的，我们这里用行向量运算来提速而不会影响训练效果
        #state_out = matmul(matmul(state_in, Utheta), U_dagger)  # 维度 [-1, 1, 2 ** n]
        state_out = matmul(matmul(state_in, Utheta), U_dagger)
        # 测量得到泡利 Z 算符的期望值 <Z>
        #E_Z = matmul(matmul(state_out, Ob),
                     #transpose(ComplexVariable(state_out.real, -state_out.imag),
                               #perm=[0, 2, 1]))
        
        # 映射 <Z> 处理成标签的估计值 
        #state_predict = E_Z.real[:, 0] * 0.5 + 0.5 + self.bias
        #loss = fluid.layers.reduce_mean((state_predict - label_pp) ** 2)
       
        #return loss, state_predict.numpy()
        return state_out.numpy()





#这里是想测试一下子分类器的输入是酉矩阵时能不能正常计算，但是报错了
with fluid.dygraph.guard():
    net = Net(n=2, depth=3, seed_paras=19)
    #inputx = rho_encode.numpy().reshape((-1,4,4))
    #print(type(inputx), inputx.shape, inputx.dtype)
    #input_data=fluid.dygraph.to_variable(rho_encode)
    inputy=(np.arange(1).reshape(-1))  
    trainy=np.asarray(inputy).astype('float64')
    #loss, state=net(state_in=input_data,label=trainy)
    state=net(state_in=rho_encode,label=trainy)
