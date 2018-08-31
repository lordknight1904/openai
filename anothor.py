import gym
import time
import tensorflow as tf
import numpy as np

EPISODES = 1000000

env_name = 'LunarLander-v2'
env = gym.make(env_name)
# print(env.observation_space.shape[0])
# print(env.action_space)
episode_observations = []
episode_rewards = []
episode_actions = []

rewards = []
eps = 0.02
epsilon = 1e-12
beta_1 = 0.9
beta_2 = 0.999
learning_rate = 0.001
reward_decay = 0.95
decay_rate = 0.99

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

layers = [obs_space, 12, 10, action_space]
weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
biases = [np.random.randn(y, 1) for y in layers[1:]]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy(predictions, targets, eps=1e-12):
    delta = np.empty([predictions.shape[0], 4])
    for index, (pre, tar) in enumerate(zip(predictions, targets)):
        p = pre.transpose()
        t = tar.transpose()
        delta[index] = -np.multiply(p[0], np.log(t[0] + eps))
    return delta


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(vector):
    vector[vector < 0] = 0
    return vector


def store_transition(s, a, r):
    episode_observations.append(s)
    episode_rewards.append(r)
    episode_actions.append(a)


def choose_actions(obs, weight, bias):
    for b, w in zip(bias, weight):
        obs = sigmoid(np.dot(w, obs) + b)
    return softmax(obs)


def calculate_future_reward(arr):
    total = 0
    gamma = 0.99
    for i in range(len(arr)):
        total += gamma ** i * arr[i]
    return total

def learn(obs, weight, bias):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    for i in range(len(episode_rewards)):
        discounted_episode_rewards[i] = calculate_future_reward(episode_rewards[i:])

    prediction = []
    target = []
    for i in episode_actions:
        prediction.append(i.transpose()[0])
        temp = np.zeros_like(i.transpose()[0])
        temp[np.argmax(i.transpose()[0])] = 1
        target.append(temp)
    neg_log_prob = np.zeros(len(prediction), dtype=object)
    for index, (pre, tar) in enumerate(zip(prediction, target)):
        neg_log_prob[index] = -np.multiply(pre, np.log(tar + epsilon))
    # print(neg_log_prob)
    delta = np.mean(discounted_episode_rewards * neg_log_prob)
    delta = delta[..., np.newaxis]
    activation = obs
    activations = [obs]
    zs = []
    for b, w in zip(bias, weight):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    nabla_b = [np.zeros(b.shape) for b in bias]
    nabla_w = [np.zeros(w.shape) for w in weight]

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, len(layers)):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    return nabla_w, nabla_b


upper_limit = 500
if __name__ == '__main__':
    for episode in range(EPISODES):
    # for episode in range(1):
        observation = env.reset()
        tic = time.clock()
        while True:
            if episode > upper_limit: env.render()
            observation = observation[..., np.newaxis]
            if np.random.uniform(0, 1) < eps:
                action = np.abs(np.random.randn(env.action_space.n))
                action = action[..., np.newaxis]
            else:
                action = choose_actions(observation, weights, biases)
            observation_, reward, done, info = env.step(np.argmax(action))
            # 4. Store transition for training
            store_transition(observation, action, reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 5:
                done = True

            if done:
                episode_rewards_sum = sum(episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)
                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Average reward: ", np.mean(rewards))
                print("Max reward so far: ", max_reward_so_far)

                # learn
                nabla_w, nabla_b = learn(observation, weights, biases)
                # temp_b = biases
                p_m = np.zeros(layers.__len__() - 1, dtype=object)
                p_v = np.zeros(layers.__len__() - 1, dtype=object)
                for index, (w, nw) in enumerate(zip(weights, nabla_w)):
                    g = nw
                    m = beta_1 * p_m[index] + (1 - beta_1) * g
                    v = beta_2 * p_v[index] + (1 - beta_2) * np.power(g, 2)
                    p_m[index] = m
                    p_v[index] = v
                    m = m / (1 - np.power(beta_1, index) + 1e-12)
                    v = v / (1 - np.power(beta_2, index) + 1e-12)
                    weights[index] = w - learning_rate * m / (np.sqrt(v) + 1e-12)

                p_m = np.zeros(layers.__len__() - 1, dtype=object)
                p_v = np.zeros(layers.__len__() - 1, dtype=object)
                for index, (b, nb) in enumerate(zip(biases, nabla_b)):
                    g = nb
                    m = beta_1 * p_m[index] + (1 - beta_1) * g
                    v = beta_2 * p_v[index] + (1 - beta_2) * np.power(g, 2)
                    p_m[index] = m
                    p_v[index] = v
                    m = m / (1 - np.power(beta_1, index) + 1e-12)
                    v = v / (1 - np.power(beta_2, index) + 1e-12)
                    biases[index] = b - learning_rate * m / (np.sqrt(v) + 1e-12)
                ##########
                episode_observations = []
                episode_rewards = []
                episode_actions = []
                break
            observation = observation_

    # choose action

    # save state|actions|state

    # if done -> learn

