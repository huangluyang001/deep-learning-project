import tensorflow as tf
import os
import gym
import random
import numpy as np
import copy
import math
#import level1.alpha_beta as ab
import time
# train on myself

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        self.gamma = 0.8
        #self.sess = tf.Session()
        pass    # 删掉这句话，并填写相应代码

    #def init_model_NN(self):

        first_dim = 128
        second_dim = 64
        learning_rate = 0.001
        # 定义自己的 网络

        self.x = tf.placeholder('float', shape=[None, 257])
        self.y_ = tf.placeholder('float', shape=[None, 1])

        self.W_1 = tf.Variable(tf.random_normal([257,first_dim],stddev=0.1/ math.sqrt(257)))
        self.b_1 = tf.Variable(tf.zeros([first_dim]))
        output_1 = tf.nn.relu(tf.matmul(self.x, self.W_1) + self.b_1)

        self.W_2 = tf.Variable(tf.random_normal([first_dim,second_dim],stddev=1./ math.sqrt(first_dim)))
        self.b_2 = tf.Variable(tf.zeros([second_dim]))
        output_2 = tf.nn.relu(tf.matmul(output_1, self.W_2) + self.b_2)
        # self.keep_prob = tf.placeholder("float")
        #h_fc1_drop = tf.nn.dropout(output_2, self.keep_prob)

        self.W_3 = tf.Variable(tf.random_normal([second_dim,1],stddev=1./ math.sqrt(second_dim)))
        self.b_3 = tf.Variable(tf.zeros([1]))
        self.y = tf.matmul(output_2, self.W_3) + self.b_3
        ones = tf.ones([1,1])



        self.cross_entropy = tf.squared_difference(self.y_,self.y)
        #self.cross_entropy = tf.reduce_mean(tf.square(self.y-self.y_))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        self.sess = tf.Session()

        # 补全代码
        # 定义初始化theta-网络
        self.xT = tf.placeholder('float', shape=[None, 257])
        self.y_T = tf.placeholder('float', shape=[None, 1])
        # self.action = tf.placeholder('float', shape=[None, 65])

        self.W_1T = tf.Variable(tf.random_normal([257, first_dim], stddev=0.1 / math.sqrt(257)))
        self.b_1T = tf.Variable(tf.zeros([first_dim]))
        output_1T = tf.nn.relu(tf.matmul(self.xT, self.W_1T) + self.b_1T)

        self.W_2T = tf.Variable(tf.random_normal([first_dim, second_dim], stddev=1. / math.sqrt(first_dim)))
        self.b_2T = tf.Variable(tf.zeros([second_dim]))
        output_2T = tf.nn.relu(tf.matmul(output_1T, self.W_2T) + self.b_2T)

        # h_fc1_dropT = tf.nn.dropout(output_2T, self.keep_prob)

        self.W_3T = tf.Variable(tf.random_normal([second_dim, 1], stddev=1. / math.sqrt(second_dim)))
        self.b_3T = tf.Variable(tf.zeros([1]))
        self.yT = tf.matmul(output_2T, self.W_3T) + self.b_3T

        self.copyTargetQNetworkOperation = [self.W_1T.assign(self.W_1), self.b_1T.assign(self.b_1),
                                            self.W_2T.assign(self.W_2), self.b_2T.assign(self.b_2),
                                            self.W_3T.assign(self.W_3), self.b_3T.assign(self.b_3)
                                            ]

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()


    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)


    def place(self, state,enables, player):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        # 删掉这句话，并填写相应代码
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            action_fake = [enables[i], player]
            q_t = self.compute_q(state, enables[i],player)
            q_list[i] = q_t
        index = np.argmax(q_list)
        r = random.random()
        if r > 0.3:
            action_new = enables[index]
        else:
            action_new = random.choice(enables)
        return action_new

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数

    def renew_weight(self,observation, action, reward, q_t,player):
        ly_x = self.encoding(observation, action,player)
        ly_x = np.reshape(ly_x,newshape=(1,257))
        y_hat = np.zeros([1,1])
        y_hat += reward + self.gamma * q_t
        W, b, y,y_,loss,_ = self.sess.run([self.W_1,self.b_1,self.y, self.y_, self.cross_entropy,self.train_step], feed_dict={self.x: ly_x,self.y_:y_hat})
        #self.sess.run(self.train_step,feed_dict={self.x: ly_x,self.y_:y_hat})
        return loss

    def compute_q(self, observation, action,player):
        ly_x = self.encoding(observation, action,player)
        ly_x = np.reshape(ly_x,newshape=(1,257))
        q = self.sess.run(self.yT,feed_dict={self.xT: ly_x})
        return q


    def encoding(self, observation, action,player):
        if player == 1:
            player = np.array(observation[1])
            opponent = np.array(observation[0])
        else:
            player = np.array(observation[0])
            opponent = np.array(observation[1])
        area = np.array(observation[2])
        if action == 65:
            action = 64
        action_vector = self.one_hot(action,65)
        return np.concatenate((np.reshape(player,64),np.reshape(opponent,64),np.reshape(area,64),action_vector),axis=-1)

    @staticmethod
    def one_hot(number,dim):
        vector = np.zeros(dim)
        vector[number] = 1.0
        return vector




def train_on_white():
    env = gym.make('Reversi8x8-v0')
    env.reset()

    agent = RL_QG_agent()
    agent.init_model_NN()
    #init = tf.initialize_all_variables()
    #agent.sess.run(init)
    agent.load_model()
    agent.copyTargetQNetwork()

    max_epochs = 200
    epoch_number = 80
    t1 = time.time()

    def policy_max(env, enables,  observation,player):
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            q_t = agent.compute_q(observation, enables[i],player)
            q_list[i] = q_t
        index = np.argmax(q_list)
        r = random.random()
        if r < 0.7:
            action_new = enables[index]
        else:
            action_new = random.choice(enables)
        fake_env = copy.deepcopy(env)
        action_fake = [enables[index], player]
        observation, reward_new, _, _ = fake_env.step(action_fake)
        q_t = agent.compute_q(observation, enables[index],player)

        return action_new, reward_new, q_t
    replay_memory = []
    for i in range(epoch_number):
        total_loss = 0
        winrate = [0, 0]

        for i_episode in range(max_epochs):
            observation = env.reset()
            observation_last = observation
            reward = 0
            action_last = [64, 1]
            observation_list = []
            reward_list = []
            action_list = []
            c = 0

            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state

            for t in range(100):
                c += 1
                if c % 10 == 0:
                    agent.copyTargetQNetwork()
                action_black = [64, 0]
                action_white = [64, 1]
                # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

                ################### 黑棋 ############################### 0表示黑棋
                #  这部分 黑棋 是随机下棋
                # env.render()  # 打印当前棋局
                enables = env.possible_actions
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1
                else:
                    action_ = random.choice(enables)
                    #action_ = agent.place(observation, enables, player=0)
                action_black[0] = action_
                action_black[1] = 0  # 黑棋 为 0
                observation, reward_2, done, info = env.step(action_black)


                ################### 白棋 ############################### 1表示白棋
                # env.render()
                enables = env.possible_actions
                # if nothing to do ,select pass
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1  # pass
                elif 65 in enables:
                    action_ = 65
                else:
                    action_, reward, q_t = policy_max(env, enables, observation, 1)  # 调用自己训练的模型

                action_white[0] = action_
                action_white[1] = 1  # 白棋 为 1
                observation_last = observation
                action_last = copy.deepcopy(action_white)
                observation_white, reward, done, info = env.step(action_white)

                observation_list.append(copy.deepcopy(observation))
                if abs(reward) == 0 and abs(reward_2) == 0:
                    reward_list=reward_2


                elif abs(reward) != 0:
                    reward_list=reward
                elif abs(reward_2) != 0:
                    reward_list=reward_2
                action_list.append(action_)
                replay_memory.append([copy.deepcopy(observation),action_white[0],copy.deepcopy(reward_list),copy.deepcopy(observation_white),copy.deepcopy(done)])
                if len(replay_memory) > 30:
                    replay_memory.pop(0)
                if len(replay_memory) == 1:
                    continue
                sample_in = random.choice(range(len(replay_memory)-1))
                if replay_memory[sample_in][4]:
                    q_t = 0
                else:
                    q_t = agent.compute_q(replay_memory[sample_in + 1][0],replay_memory[sample_in+1][1], player=1)
                loss = agent.renew_weight(replay_memory[sample_in][0], replay_memory[sample_in][1],
                                          -replay_memory[sample_in][2], q_t, player=1)
                total_loss += loss

                if done:  # 游戏 结束
                    #print("Episode finished after {} timesteps".format(t + 1))
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    if black_score > 32:
                        #print("黑棋赢了！")
                        winrate[0] += 1
                    else:
                        #print("白棋赢了！")
                        winrate[1] += 1
                    #print(black_score, r_t, reward_2, reward, action_last)
                    break
            #assert len(reward_list) == len(observation_list)
            #assert len(observation_list) == len(action_list)
            #length = len(observation_list)
            #for i in range(len(reward_list)):
                #j = length - i - 1
                # if j + 1 == length:
                #    q_t = 0
                # else:
                   #q_t = agent.compute_q(observation_list[j + 1],action_list[j + 1],player=1)
                #loss = agent.renew_weight(observation_list[j],action_list[j], - reward_list[j], q_t,player=1)
                #total_loss += loss


        print(winrate)
        total_loss /= max_epochs
        t2 = time.time()
        print('Time:' ,t2-t1)
        print('Total Loss:', total_loss)
        agent.save_model()

def train_on_black():
    env = gym.make('Reversi8x8-v0')
    env.reset()

    agent = RL_QG_agent()
    agent.init_model_NN()
    # init = tf.initialize_all_variables()
    # agent.sess.run(init)
    agent.load_model()
    agent.copyTargetQNetwork()

    max_epochs = 200
    epoch_number = 80
    t1 = time.time()

    def policy_max(env, enables, observation, player):
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            q_t = agent.compute_q(observation, enables[i], player)
            q_list[i] = q_t
        index = np.argmax(q_list)
        r = random.random()
        if r < 0.7:
            action_new = enables[index]
        else:
            action_new = random.choice(enables)
        fake_env = copy.deepcopy(env)
        action_fake = [enables[index], player]
        observation, reward_new, _, _ = fake_env.step(action_fake)
        q_t = agent.compute_q(observation, enables[index], player)

        return action_new, reward_new, q_t

    replay_memory = []
    for i in range(epoch_number):
        total_loss = 0
        winrate = [0, 0]

        for i_episode in range(max_epochs):
            observation = env.reset()
            observation_last = observation
            reward = 0
            action_last = [64, 1]
            observation_list = []
            reward_list = []
            action_list = []
            c = 0

            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state

            for t in range(100):
                c += 1
                if c % 10 == 0:
                    agent.copyTargetQNetwork()
                action_black = [64, 0]
                action_white = [64, 1]
                # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）


                ################### 白棋 ############################### 1表示白棋
                # env.render()
                observation2 = copy.deepcopy(observation)
                enables = env.possible_actions
                # if nothing to do ,select pass
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1  # pass
                elif 65 in enables:
                    action_ = 65
                else:
                    action_, reward, q_t = policy_max(env, enables, observation, 0)  # 调用自己训练的模型

                action_black[0] = action_
                action_black[1] = 0  # 白棋 为 1
                observation_last = observation
                action_last = copy.deepcopy(action_black)
                observation_black, reward, done, info = env.step(action_black)


                ################### 黑棋 ############################### 0表示黑棋
                #  这部分 白棋 是随机下棋
                # env.render()  # 打印当前棋局
                enables = env.possible_actions
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1
                else:
                    action_ = random.choice(enables)
                    # action_ = agent.place(observation, enables, player=0)
                action_white[0] = action_
                action_white[1] = 1  # 黑棋 为 0
                observation, reward_2, done, info = env.step(action_white)

                observation_list.append(copy.deepcopy(observation2))
                if abs(reward) == 0 and abs(reward_2) == 0:
                    reward_list = reward_2
                elif abs(reward) != 0:
                    reward_list = reward
                elif abs(reward_2) != 0:
                    reward_list = reward_2

                action_list.append(action_)
                replay_memory.append([copy.deepcopy(observation2), action_black[0], copy.deepcopy(reward_list),
                                      copy.deepcopy(observation_black), copy.deepcopy(done)])

                if len(replay_memory) > 30:
                    replay_memory.pop(0)
                if len(replay_memory) == 1:
                    continue
                sample_in = random.choice(range(len(replay_memory) - 1))
                if replay_memory[sample_in][4]:
                    q_t = 0
                else:
                    q_t = agent.compute_q(replay_memory[sample_in + 1][0], replay_memory[sample_in + 1][1], player=0)
                loss = agent.renew_weight(replay_memory[sample_in][0], replay_memory[sample_in][1],
                                          replay_memory[sample_in][2], q_t, player=0)
                total_loss += loss

                if done:  # 游戏 结束
                    # print("Episode finished after {} timesteps".format(t + 1))
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    if black_score > 32:
                        # print("黑棋赢了！")
                        winrate[0] += 1
                    else:
                        # print("白棋赢了！")
                        winrate[1] += 1
                    # print(black_score, r_t, reward_2, reward, action_last)
                    break
                    # assert len(reward_list) == len(observation_list)
                    # assert len(observation_list) == len(action_list)
                    # length = len(observation_list)
                    # for i in range(len(reward_list)):
                    # j = length - i - 1
                    # if j + 1 == length:
                    #    q_t = 0
                    # else:
                    # q_t = agent.compute_q(observation_list[j + 1],action_list[j + 1],player=1)
                    # loss = agent.renew_weight(observation_list[j],action_list[j], - reward_list[j], q_t,player=1)
                    # total_loss += loss

        print(winrate)
        total_loss /= max_epochs
        t2 = time.time()
        print('Time:', t2 - t1)
        print('Total Loss:', total_loss)
        agent.save_model()


if __name__ == '__main__':
    # train_on_white()
    train_on_black()