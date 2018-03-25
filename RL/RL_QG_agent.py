import tensorflow as tf
import os
import gym
import random
import numpy as np
import copy
import math
import level1.alpha_beta as ab
import time

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        self.gamma = 0.5
        pass    # 删掉这句话，并填写相应代码

    def init_model_NN(self):


        first_dim = 128
        second_dim = 64
        learning_rate = 0.001
        # 定义自己的 网络

        self.x = tf.placeholder('float', shape=[None, 257])
        self.y_ = tf.placeholder('float', shape=[None, 1])

        self.W_1 = tf.Variable(tf.random_normal([257,first_dim],stddev=1./ math.sqrt(257)))
        self.b_1 = tf.Variable(tf.zeros([first_dim]))
        output_1 = tf.nn.relu(tf.matmul(self.x, self.W_1) + self.b_1)

        W_2 = tf.Variable(tf.random_normal([first_dim,second_dim],stddev=1./ math.sqrt(first_dim)))
        b_2 = tf.Variable(tf.zeros([second_dim]))
        output_2 = tf.nn.relu(tf.matmul(output_1, W_2) + b_2)

        W_3 = tf.Variable(tf.random_normal([second_dim,1],stddev=1./ math.sqrt(second_dim)))
        b_3 = tf.Variable(tf.zeros([1]))
        self.y = tf.nn.sigmoid(tf.matmul(output_2, W_3) + b_3)
        ones = tf.ones([1,1])    



        self.cross_entropy = tf.squared_difference(self.y_,self.y)
        #self.cross_entropy = - tf.reduce_mean(tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y,1e-10,1.0)))) - tf.reduce_mean(tf.reduce_sum((ones-self.y_)*tf.log(tf.clip_by_value((ones-self.y),1e-10,1.0))))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

        # 补全代码






    def place(self,state,enables, player, env):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        # 删掉这句话，并填写相应代码
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            fake_env = copy.deepcopy(env)
            action_fake = [enables[i], player]
            observation, _, _, _ = fake_env.step(action_fake)
            q_t = self.compute_q(observation, enables[i])
            q_list[i] = q_t
        index = np.argmax(q_list)
        action_new = enables[index]

        return action_new

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数

    def renew_weight(self,observation, action, reward, q_t):
        ly_x = self.encoding(observation, action)
        ly_x = np.reshape(ly_x,newshape=(1,257))
        y_hat = np.zeros([1,1])
        y_hat += reward + self.gamma * q_t
        if y_hat < 0:
            y_hat = np.zeros([1,1])
        if y_hat > 1:
            y_hat = np.ones([1,1])
        W, b, y,y_,loss,_ = self.sess.run([self.W_1,self.b_1,self.y, self.y_, self.cross_entropy,self.train_step], feed_dict={self.x: ly_x,self.y_:y_hat})
        #self.sess.run(self.train_step,feed_dict={self.x: ly_x,self.y_:y_hat})
        return loss

    def compute_q(self, observation, action):
        ly_x = self.encoding(observation, action)
        ly_x = np.reshape(ly_x,newshape=(1,257))
        q = self.sess.run(self.y,feed_dict={self.x: ly_x})
        return q


    def encoding(self, observation, action):
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
    #agent.load_model()

    max_epochs = 300
    epoch_number = 100
    t1 = time.time()

    def policy_max(env, enables, player, observation):
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            q_t = agent.compute_q(observation, enables[i])
            q_list[i] = q_t
        index = np.argmax(q_list)
        action_new = enables[index]
        fake_env = copy.deepcopy(env)
        action_fake = [enables[index], player]
        observation, reward_new, _, _ = fake_env.step(action_fake)
        q_t = agent.compute_q(observation, enables[index])

        return action_new, reward_new, q_t

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


            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state

            for t in range(100):
                action_black = [64, 0]
                action_white = [64, 1]
                # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

                ################### 黑棋 ############################### 0表示黑棋
                #  这部分 黑棋 是随机下棋
                #env.render()  # 打印当前棋局
                enables = env.possible_actions
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1
                else:
                    action_ = random.choice(enables)
                action_black[0] = action_
                action_black[1] = 0  # 黑棋 为 0
                observation, reward_2, done, info = env.step(action_black)




                ################### 白棋 ############################### 1表示白棋
                #env.render()
                enables = env.possible_actions
                # if nothing to do ,select pass
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1  # pass
                elif 65 in enables:
                    action_ = 65
                else:
                    action_, reward, q_t = policy_max(env, enables, 1, observation)  # 调用自己训练的模型

                action_white[0] = action_
                action_white[1] = 1  # 白棋 为 1
                observation_last = observation
                action_last = copy.deepcopy(action_white)
                observation_white, reward, done, info = env.step(action_white)

                observation_list.append(copy.deepcopy(observation))
                if abs(reward) == 0 and abs(reward_2) == 0:
                    reward_list.append(reward_2)
                elif abs(reward) != 0:
                    reward_list.append(reward)
                elif abs(reward_2) != 0:
                    reward_list.append(reward_2)

                action_list.append(action_ )



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
            assert len(reward_list) == len(observation_list)
            assert len(observation_list) == len(action_list)
            length = len(observation_list)
            for i in range(len(reward_list)):
                j = length - i - 1
                if j + 1 == length:
                    q_t = 0
                else:
                    q_t = agent.compute_q(observation_list[j + 1],action_list[j + 1])
                loss = agent.renew_weight(observation_list[j],action_list[j], - reward_list[j], q_t)
                total_loss += loss


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
    init = tf.global_variables_initializer()
    agent.sess.run(init)
    #agent.load_model()

    max_epochs = 100
    epoch_number = 100

    def policy_max(env, enables, player, observation):
        q_list = np.zeros(len(enables))
        for i in range(len(enables)):
            q_t = agent.compute_q(observation, enables[i])
            q_list[i] = q_t
        index = np.argmax(q_list)
        action_new = enables[index]
        fake_env = copy.deepcopy(env)
        action_fake = [enables[index], player]
        observation, reward_new, _, _ = fake_env.step(action_fake)
        q_t = agent.compute_q(observation, enables[index])

        return action_new, reward_new, q_t

    for i in range(epoch_number):
        total_loss = 0
        winrate = [0, 0]

        for i_episode in range(max_epochs):
            observation = env.reset()
            observation_last = observation
            reward = 0
            action_last = [64, 1]

            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state

            for t in range(100):
                action_black = [64, 0]
                action_white = [64, 1]
                # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

                ################### 黑棋 ############################### 0表示黑棋
                #  这部分 黑棋 是随机下棋
                #env.render()  # 打印当前棋局
                enables = env.possible_actions
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1
                elif 65 in enables:
                    action_ = 65
                    if abs(reward_2) > 1e-4:
                        r_t = reward_2
                    elif abs(reward) > 1e-4:
                        r_t = reward
                        q_t = 0.0
                else:
                    action_, r_t, q_t = policy_max(env, enables, 1, observation)
                action_black[0] = action_
                action_black[1] = 0  # 黑棋 为 0
                observation, reward_2, done, info = env.step(action_black)



                ################### 白棋 ############################### 1表示白棋
                #env.render()
                enables = env.possible_actions
                # if nothing to do ,select pass
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1  # pass

                else:
                    action_ = random.choice(enables)

                if action_last[0] != 65:
                    r_t = - r_t
                    loss = agent.renew_weight(observation_last, action_last, r_t, q_t)
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
                    total_loss = total_loss / (i_episode + 1)
                    #print('Total loss:',total_loss)
                    break
                action_white[0] = action_
                action_white[1] = 1  # 白棋 为 1
                observation_last = observation
                action_last = copy.deepcopy(action_white)
                observation_white, reward, done, info = env.step(action_white)


        print(winrate)
        total_loss /= max_epochs
        print('Total Loss:', total_loss)
        agent.save_model()


if __name__ == '__main__':
    train_on_white()