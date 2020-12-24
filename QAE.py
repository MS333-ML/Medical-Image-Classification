import numpy as np
from numpy import diag
import scipy

import paddle
from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle.complex import matmul, trace, kron
from paddle_quantum.utils import dagger, state_fidelity, partial_trace


N_A = 2        # 系统 A 的量子比特数
N_B = 1        # 系统 B 的量子比特数
N = N_A + N_B  # 总的量子比特数

scipy.random.seed(1)                            # 固定随机种子
V = scipy.stats.unitary_group.rvs(2**N)         # 随机生成一个酉矩阵
#D = diag([0.4, 0.2, 0.2, 0.1, -0.1, 0, 0, 0])    # 输入目标态rho的谱
D = diag([ 0.87121176, -0.49674604,  0.49909047,  0.39252172, 
          -0.30386159, 0.10832246, -0.05726166, -0.01327711]) #前四个最大的 0.87+0.49+0.39+0.11 = 1.86
D = diag([0.87121176,0.49909047,0.39252172,0.10832246,-0.01327711, -0.05726166,-0.30386159,-0.49674604])
V_H = V.conj().T                                # 进行厄尔米特转置
rho_in = (V @ D @ V_H).astype('complex128')      # 生成 rho_in

# 将 C 量子系统初始化为
rho_C = np.diag([1,0]).astype('complex128')

# ## 搭建量子神经网络
# 
# 在这里，我们用量子神经网络来作为编码器和解码器。假设系统 $A$ 有 $N_A$ 个量子比特，系统 $B$ 和 $C$ 分别有$N_B$ 个量子比特，量子神经网络的深度为 D。编码器 $E$ 作用在系统 $A$ 和 $B$ 共同构成的总系统上，解码器 $D$ 作用在$A$ 和 $C$ 共同构成的总系统上。在我们的问题里，$N_{A} = 2$，$N_{B} = 1$。


# 设置电路参数
cir_depth = 8                        # 电路深度
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


# 超参数设置
N_A = 2        # 系统 A 的量子比特数
N_B = 1        # 系统 B 的量子比特数
N = N_A + N_B  # 总的量子比特数
LR = 0.2       # 设置学习速率
ITR = 1000      # 设置迭代次数
SEED = 14      # 固定初始化参数用的随机数种子

class NET(fluid.dygraph.Layer):
    """
    Construct the model net
    """
    def __init__(self, shape, param_attr=fluid.initializer.Uniform(
        low=0.0, high=2 * np.pi, seed = SEED), dtype='float64'):
        super(NET, self).__init__()
        
        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        self.rho_in = fluid.dygraph.to_variable(rho_in)
        self.rho_C = fluid.dygraph.to_variable(rho_C)
        self.theta = self.create_parameter(shape=shape, 
                     attr=param_attr, dtype=dtype, is_bias=False)
    
    # 定义损失函数和前向传播机制
    def forward(self):
        # 生成初始的编码器 E 和解码器 D\n",
        E = Encoder(self.theta)
        E_dagger = dagger(E)
        D = E_dagger
        D_dagger = E

        # 编码量子态 rho_in
        rho_BA = matmul(matmul(E, self.rho_in), E_dagger)
        
        # 取 partial_trace() 获得 rho_encode 与 rho_trash
        rho_encode = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 1)
        rho_trash = partial_trace(rho_BA, 2 ** N_B, 2 ** N_A, 2)

        # 解码得到量子态 rho_out
        rho_CA = kron(self.rho_C, rho_encode)
        rho_out = matmul(matmul(D, rho_CA), D_dagger)
        
        # 通过 rho_trash 计算损失函数
        
        zero_Hamiltonian = fluid.dygraph.to_variable(np.diag([1,0]).astype('complex128'))
        loss = 1 - (trace(matmul(zero_Hamiltonian, rho_trash))).real

        return loss, self.rho_in, rho_out


# 初始化paddle动态图机制  
with fluid.dygraph.guard():

    # 生成网络
    net = NET([theta_size])

    # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMS prop.
    opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR, 
                          parameter_list=net.parameters())

    # 优化循环
    for itr in range(1, ITR + 1):
        
        #  前向传播计算损失函数
        loss, rho_in, rho_out = net()
        
        # 在动态图机制下，反向传播极小化损失函数
        loss.backward()
        opt.minimize(loss)
        net.clear_gradients()
        
        # 计算并打印保真度
        fid = state_fidelity(rho_in.numpy(), rho_out.numpy())

        if itr % 10 == 0:
            print('iter:', itr, 'loss:', '%.4f' % loss, 'fid:', '%.4f' % np.square(fid))
