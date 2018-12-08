import numpy as np
import math
import random
import copy
from numpy.matlib import rand
from numpy.random.mtrand import random_sample
import time

#sigmoid 函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#遗传算法个体类
class GAIndividual:
    def setmatrix(self,matrix):
        self.matrix=matrix

    def __init__(self):
        self.fitness=0

    #随机生成染色体
    def generate(self):
        matrix = 8*random_sample((3,8))-4
        for i in range(3):
            for j in range(8):
                if i + 6 == j:
                    matrix[i][j] = 0
        return matrix
    # 计算染色体的适应性函数
    def calculateFitness(self):
        matrix1=self.matrix
        nn=BPNN(matrix1)
        nn.train()
        pattern = []
        for j in range(32):
            inputs = [-1] + list(map(int, bin(j)[2:].rjust(5, '0')))
            pattern.append(inputs)
        self.fitness=nn.test(pattern)
        #print(self.fitness)
        return self.fitness
#定义 BPNN 神经网络的类
class BPNN:
    #初始化
    def __init__(self,matrix):
        #定义输入节点数
        self.input_size = 6
        #随机生成节点数
        #self.node_sizes=np.random.randint(self.input_size,20)
        self.node_sizes = 9
        #随机生成权重矩阵 只存储非输入节点与非输出结点之间的weights
        self.matrix=matrix
        #self.matrix = 12*random_sample((self.node_sizes - self.input_size, self.node_sizes - 1))-6
        #self.matrix=np.random.normal(loc=2, scale=10, size=(self.node_sizes - self.input_size, self.node_sizes - 1))


    #实现前馈功能
    def predict(self,inputs):
        # 定义一个一维数组存放所有节点的output
        # 输入节点及偏置节点的input就是其output
        self.node_outputs=[]
        for i in range(self.input_size):
            self.node_outputs.append(inputs[i])
        #计算其他非输出节点的output
        for i in range(self.node_sizes-self.input_size):
            sum=0
            for j in range(len(self.node_outputs)):
                sum += self.matrix[i][j]*self.node_outputs[j]
            sum=sigmoid(sum)
            self.node_outputs.append(sum)
        #得到输出节点的output
        return self.node_outputs[self.node_sizes-1]

    # #计算误差值
    # def error(self,case):
    #     self.predict(case)
    #     # 首先算出这个case的正确输出 expect
    #     sum_1 = 0
    #     for i in range(1, len(case)):
    #         if case[i] == 1:
    #             sum_1 += 1
    #     if sum_1 % 2 == 0:
    #         expect = 1
    #     else:
    #         expect = 0
    #     return  1 / 2 * pow((self.predict(case) - expect), 2)


    #bp算法
    def backforward(self,case,learning_rate):
            # 使case通过前馈神经网络得到输出值
            self.predict(case)
            # 首先算出这个case的正确输出 expect
            sum_1=0
            for i in range(1,len(case)):
                if case[i]==1:
                    sum_1+=1
            if sum_1%2==0:
                expect=1
            else:
                expect=0
            # 算出输出层的误差值
            error=1/2*pow((self.node_outputs[self.node_sizes-1]-expect),2)
            # 算出最后的误差对每条权重的改变值，存在矩阵derivative中
            derivative = np.zeros((self.node_sizes - self.input_size, self.node_sizes - 1))
            # 每个节点总误差对intput的偏导存储在一个数组 node_derivative__output[]
            node_derivative__output = np.zeros(self.node_sizes)
            # 求节点intput对权重的偏导
            def derivative_input_weights(i, j):
                return  self.node_outputs[j]
            # 求总误差对intput的偏导
            def derivative_error_input(i):
                if i == self.node_sizes - self.input_size - 1:
                    return (self.node_outputs[self.node_sizes-1]-expect)*(1 - self.node_outputs[i + self.input_size]) * self.node_outputs[i + self.input_size]
                else:
                    sum_derivative = 0
                    for h in range(i+1, self.node_sizes - self.input_size):
                        if self.matrix[h][i+self.input_size] != 0:
                            sum_derivative += node_derivative__output[h+self.input_size] * self.matrix[h][i+self.input_size]*(1 - self.node_outputs[i + self.input_size]) * self.node_outputs[i + self.input_size]
                    return sum_derivative

            #计算每条边的改变值
            for i in range(self.node_sizes-self.input_size-1,-1,-1):
                for j in range(self.node_sizes-1):
                       derivative[i][j]=derivative_error_input(i)*derivative_input_weights(i,j)*learning_rate*(-1)
                node_derivative__output[i+self.input_size]=derivative_error_input(i)
            return derivative
            #  更新权值矩阵

    def update_matrix(self, derivative):
        for i in range(self.node_sizes - self.input_size):
            for j in range(self.node_sizes - 1):
                if self.matrix[i][j] != 0:
                    self.matrix[i][j] += derivative[i][j]
                else:
                    self.matrix[i][j] = 0
        # return 更新过的权重矩阵
        # return self.matrix

        # 用一些test cases 来训练这个神经网络 并返回权重矩阵

    def train(self, iterations=50):
        pattern = []
        deviation = np.zeros((self.node_sizes - self.input_size, self.node_sizes - 1))
        for i in range(iterations):
            for j in range(32):
                inputs = [-1] + list(map(int, bin(j)[2:].rjust(5, '0')))
                pattern.append(inputs)
                self.backforward(pattern[j], 0.1)
                deviation += self.backforward(pattern[j], 0.1)
            self.update_matrix(deviation)
            # print(self.matrix)

    #测试神经网络在测试集上的正确个数
    def test(self, pattern):
        n=0
        for p in range(32):
            inputs = pattern[p]
            sum_1 = 0
            for i in range(1,len(inputs)):
                 if inputs[i]==1:
                     sum_1+=1
            if sum_1%2==0:
                 expect=1
            else:
                 expect=0
            if expect==np.around(self.predict(inputs)):
                  n+=1

        return n




    # def train(self,pattern):
    #     alpha = 0.1
    #     beta_1 = 0.5
    #     beta_2 = 0.999  # initialize the values of the parameters
    #     epsilon = 1e-8
    #
    #     m_t = np.zeros(shape=self.matrix.shape)
    #     v_t = np.zeros(shape=self.matrix.shape)
    #     deviation = np.zeros((self.node_sizes - self.input_size, self.node_sizes - 1))
    #     for t in range(1,1000):
    #         for p in pattern:
    #             deviation += self.backforward(p)
    #         g_t = deviation  # computes the gradient of the stochastic function
    #         m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
    #         v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
    #         m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
    #         v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
    #         self.matrix -= (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
    #
    #
    #         #print(self.matrix)




#遗传算法类
class GeneticAlgorithm:
    #初始化
    def __init__(self,sizepop,MAXGEN):
        self.sizepop=sizepop
        self.MAXGEN=MAXGEN
        self.population=[]
        self.fitness=np.zeros(sizepop)
        self.iteration = 50

    #初始化种群
    def initialize(self):
        for i in range(0,self.sizepop):
            ind=GAIndividual()

            ind.setmatrix(ind.generate())
            self.population.append(ind)

    # 评估种群的适应性函数值
    def evaluate(self):
        sum=0
        for i in range(self.sizepop):

            self.fitness[i]=self.population[i].calculateFitness()
           # print(self.fitness[i])
    # 遗传算法
    def solve(self):
        self.t=0
        self.initialize()
        self.evaluate()
        best=np.max(self.fitness)
        newpop1=self.population
        bestIndex=np.argmax(self.fitness)
        self.best=copy.deepcopy(self.population[bestIndex])
        while (self.t<self.MAXGEN and best!=32):
            self.t+=1
            newpop=[]
            arr=[]
            fitness_sum=[]
            population_sum=[]
            for i in range(self.sizepop):
                bestmatrix = self.best.matrix
                bestmatrix1= 8*random_sample((3,8))-4
                ind = GAIndividual()
                ind.matrix=bestmatrix1
                newpop.append(ind)
            population_sum=newpop1+ newpop
            for i in range(2*self.sizepop):
                fitness_sum.append(population_sum[i].calculateFitness())
            fitness_sum= np.array(fitness_sum)
            arr=np.argsort(-fitness_sum)
            for i in range(self.sizepop):
                newpop[i]=population_sum[arr[i]]
            self.population= []
            self.population = newpop
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            best = np.max(self.fitness)
            newpop1 = self.population
            if best==32:
                break
        matrix2=self.population[bestIndex].matrix


        for i in range(3):
            for j in range(8):
                print(matrix2[i][j],end=' ')
            print("")
        print(self.iteration)




begin=time.time()
ge=GeneticAlgorithm(5,5000)
ge.solve()
