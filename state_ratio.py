import numpy as np
import tensorflow as tf
from time import sleep
import sys

# Hyper parameter
# Learning_rate = 1e-3
# initial_stddev = 0.5

# Training Parameter
training_batch_size = 512
training_maximum_iteration = 3001
TEST_NUM = 2000


class Density_Ratio_kernel(object):
    def __init__(self, obs_dim, w_hidden, Learning_rate, reg_weight):
        # place holder
        self.state = tf.placeholder(tf.float32, [None, obs_dim])
        self.med_dist = tf.placeholder(tf.float32, [])
        self.next_state = tf.placeholder(tf.float32, [None, obs_dim])

        self.state2 = tf.placeholder(tf.float32, [None, obs_dim])
        self.next_state2 = tf.placeholder(tf.float32, [None, obs_dim])
        self.policy_ratio = tf.placeholder(tf.float32, [None])
        self.policy_ratio2 = tf.placeholder(tf.float32, [None])

        # density ratio for state and next state
        w = self.state_to_w(self.state, obs_dim, w_hidden)
        w_next = self.state_to_w(self.next_state, obs_dim, w_hidden)
        w2 = self.state_to_w(self.state2, obs_dim, w_hidden)
        w_next2 = self.state_to_w(self.next_state2, obs_dim, w_hidden)
        norm_w = tf.reduce_mean(w)
        norm_w_next = tf.reduce_mean(w_next)
        norm_w_beta = tf.reduce_mean(w * self.policy_ratio)
        norm_w2 = tf.reduce_mean(w2)
        norm_w_next2 = tf.reduce_mean(w_next2)
        norm_w_beta2 = tf.reduce_mean(w2 * self.policy_ratio2)
        self.output = w

        # calculate loss function
        # x = w * self.policy_ratio - w_next
        # x2 = w2 * self.policy_ratio2 - w_next2
        # x = w * self.policy_ratio / norm_w_beta - w_next / norm_w
        # x2 = w2 * self.policy_ratio2 / norm_w_beta2 - w_next2 / norm_w2
        x = w * self.policy_ratio / norm_w - w_next / norm_w_next
        x2 = w2 * self.policy_ratio2 / norm_w2 - w_next2 / norm_w_next2

        diff_xx = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.next_state2, 0)
        K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx), axis=-1) / (2.0 * self.med_dist * self.med_dist))
        norm_K = tf.reduce_sum(K_xx)

        loss_xx = tf.matmul(tf.matmul(tf.expand_dims(x, 0), K_xx), tf.expand_dims(x2, 1))

        # self.loss = tf.squeeze(loss_xx)/(norm_w*norm_w2*norm_K)
        self.loss = tf.squeeze(loss_xx) / norm_K
        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
        self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss + reg_weight * self.reg_loss)

        # Debug
        self.debug1 = tf.reduce_mean(w)
        self.debug2 = tf.reduce_mean(w_next)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def close_Session(self):
        tf.reset_default_graph()
        self.sess.close()

    def state_to_w(self, state, obs_dim, hidden_dim_dr):
        with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
            w = tf.ones([tf.shape(state)[0]])
            for i in range(obs_dim / 4):
                w_part_i = self.state_to_w_tl(state[:, i:(i + 4)], 4, hidden_dim_dr)
                w = w * w_part_i
            return w

    def state_to_w_tl(self, state, obs_dim, hidden_dim_dr):
        with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
            # First layer
            W1 = tf.get_variable('W1', initializer=tf.random_normal(
                shape=[obs_dim, hidden_dim_dr]))  # , regularizer = tf.contrib.layers.l2_regularizer(1.))
            b1 = tf.get_variable('b1', initializer=tf.zeros(
                [hidden_dim_dr]))  # , regularizer = tf.contrib.layers.l2_regularizer(1.))
            z1 = tf.matmul(state, W1) + b1
            mean_z1, var_z1 = tf.nn.moments(z1, [0])
            scale_z1 = tf.get_variable('scale_z1', initializer=tf.ones([hidden_dim_dr]))
            beta_z1 = tf.get_variable('beta_z1', initializer=tf.zeros([hidden_dim_dr]))
            l1 = tf.tanh(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))

            # Second layer
            W2 = tf.get_variable('W2', initializer=0.01 * tf.random_normal(shape=[hidden_dim_dr, 1]),
                                 regularizer=tf.contrib.layers.l2_regularizer(1.))
            b2 = tf.get_variable('b2', initializer=tf.zeros([1]), regularizer=tf.contrib.layers.l2_regularizer(1.))
            z2 = tf.matmul(l1, W2) + b2
            # return tf.exp(tf.squeeze(z2))
            # mean_z2, var_z2 = tf.nn.moments(z2, [0])
            # scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([1]))
            # beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([1]))
            # l2 = tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10)
            return tf.log(1 + tf.exp(tf.squeeze(z2)))

    def get_density_ratio(self, states):
        return self.sess.run(self.output, feed_dict={
            self.state: states
        })

    def train(self, SASR, policy0, policy1, batch_size=training_batch_size, max_iteration=training_maximum_iteration,
              test_num=TEST_NUM, fPlot=False, epsilon=1e-3):
        S = []
        SN = []
        # POLICY_RATIO = []
        # POLICY_RATIO2 = []
        PI1 = []
        PI0 = []
        REW = []
        for sasr in SASR:
            for state, action, next_state, reward in sasr:
                # POLICY_RATIO.append((epsilon + policy1.pi(state, action))/(epsilon + policy0.pi(state, action)))
                # POLICY_RATIO2.append(policy1.pi(state, action)/policy0.pi(state, action))
                # POLICY_RATIO.append(epsilon + (1-epsilon) * policy1.pi(state, action)/policy0.pi(state, action))
                PI1.append(policy1.pi(state, action))
                PI0.append(policy0.pi(state, action))
                S.append(state)
                SN.append(next_state)
                REW.append(reward)
        # normalized

        S = np.array(S)
        S_max = np.max(S, axis=0)
        S_min = np.min(S, axis=0)
        S = (S - S_min) / (S_max - S_min)
        SN = (np.array(SN) - S_min) / (S_max - S_min)

        if test_num > 0:
            S_test = np.array(S[:test_num])
            SN_test = np.array(SN[:test_num])
            # POLICY_RATIO_test = np.array(POLICY_RATIO[:test_num])
            PI1_test = np.array(PI1[:test_num])
            PI0_test = np.array(PI0[:test_num])

        S = np.array(S[test_num:])
        SN = np.array(SN[test_num:])
        # POLICY_RATIO = np.array(POLICY_RATIO[test_num:])
        # POLICY_RATIO2 = np.array(POLICY_RATIO2[test_num:])
        PI1 = np.array(PI1[test_num:])
        PI0 = np.array(PI0[test_num:])
        REW = np.array(REW[test_num:])
        N = S.shape[0]

        subsamples = np.random.choice(N, 1000)
        s = S[subsamples]
        med_dist = np.median(np.sqrt(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis=-1)))

        for i in range(max_iteration):
            if test_num > 0 and i % 500 == 0:
                subsamples = np.random.choice(test_num, batch_size)
                s_test = S_test[subsamples]
                sn_test = SN_test[subsamples]
                # policy_ratio_test = POLICY_RATIO_test[subsamples]
                policy_ratio_test = (PI1_test[subsamples] + epsilon) / (PI0_test[subsamples] + epsilon)

                subsamples = np.random.choice(test_num, batch_size)
                s_test2 = S_test[subsamples]
                sn_test2 = SN_test[subsamples]
                # policy_ratio_test2 = POLICY_RATIO_test[subsamples]
                policy_ratio_test2 = (PI1_test[subsamples] + epsilon) / (PI0_test[subsamples] + epsilon)

                test_loss, reg_loss, norm_w, norm_w_next = self.sess.run(
                    [self.loss, self.reg_loss, self.debug1, self.debug2], feed_dict={
                        self.med_dist: med_dist,
                        self.state: s_test,
                        self.next_state: sn_test,
                        self.policy_ratio: policy_ratio_test,
                        self.state2: s_test2,
                        self.next_state2: sn_test2,
                        self.policy_ratio2: policy_ratio_test2
                    })
                print('----Iteration = {}-----'.format(i))
                print("Testing error = {}".format(test_loss))
                print('Regularization loss = {}'.format(reg_loss))
                print('Norm_w = {}'.format(norm_w))
                print('Norm_w_next = {}'.format(norm_w_next))
                DENR = self.get_density_ratio(S)
                # T = DENR*POLICY_RATIO2
                T = DENR * PI1 / PI0
                print('DENR = {}'.format(np.sum(T * REW) / np.sum(T)))
                sys.stdout.flush()
            # epsilon *= 0.9

            subsamples = np.random.choice(N, batch_size)
            s = S[subsamples]
            sn = SN[subsamples]
            # policy_ratio = POLICY_RATIO[subsamples]
            policy_ratio = (PI1[subsamples] + epsilon) / (PI0[subsamples] + epsilon)

            # subsamples = np.random.choice(N, batch_size)
            s2 = S[subsamples]
            sn2 = SN[subsamples]
            # policy_ratio2 = POLICY_RATIO[subsamples]
            policy_ratio2 = (PI1[subsamples] + epsilon) / (PI0[subsamples] + epsilon)

            self.sess.run(self.train_op, feed_dict={
                self.med_dist: med_dist,
                self.state: s,
                self.next_state: sn,
                self.policy_ratio: policy_ratio,
                self.state2: s2,
                self.next_state2: sn2,
                self.policy_ratio2: policy_ratio2
            })
        DENR = self.get_density_ratio(S)
        # T = DENR*POLICY_RATIO2
        T = DENR * PI1 / PI0
        return np.sum(T * REW) / np.sum(T)

    def evaluate(self, SASR0, policy0, policy1):
        S = []
        POLICY_RATIO = []
        REW = []
        for sasr in SASR0:
            for state, action, next_state, reward in sasr:
                POLICY_RATIO.append(policy1.pi(state, action) / policy0.pi(state, action))
                S.append(state)
                REW.append(reward)

        S = np.array(S)
        S_max = np.max(S, axis=0)
        S_min = np.min(S, axis=0)
        S = (S - S_min) / (S_max - S_min)
        POLICY_RATIO = np.array(POLICY_RATIO)
        REW = np.array(REW)
        DENR = self.get_density_ratio(S)
        T = DENR * POLICY_RATIO
        return np.sum(T * REW) / np.sum(T)