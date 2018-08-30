import gym
import time
import tensorflow as tf
import numpy as np

EPISODES = 100000

env_name = 'LunarLander-v2'
env = gym.make(env_name)
# print(env.observation_space.shape[0])
# print(env.action_space)
episode_observations = []
episode_rewards = []
episode_actions = []

rewards = []
eps = 0.02

learning_rate = 0.02
reward_decay = 0.99
decay_rate = 0.95

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
    return obs.transpose()[0]


def learn(obs, weight, bias):
    a = obs
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * reward_decay + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)

    nabla_b = [np.zeros(b.shape) for b in bias]
    nabla_w = [np.zeros(w.shape) for w in weight]

    activation = a
    activations = [a]
    zs = []
    for b, w in zip(bias, weight):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    hot_vector = np.zeros_like(episode_actions)
    for index, ep in enumerate(episode_actions, start=0):
        hot_vector[index][np.argmax(episode_actions[index])] = 1
    neg_log_prob = cross_entropy(hot_vector, episode_actions)
    delta = neg_log_prob.transpose() * discounted_episode_rewards
    delta = np.sum(delta, axis=1)
    delta = delta[:, np.newaxis]

    print(delta.shape)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, len(layers)):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return nabla_w, nabla_b


upper_limit = 200
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
            else:
                action = choose_actions(observation, weights, biases)
            observation_, reward, done, info = env.step(np.random.choice(env.action_space.n, p=softmax(action)))
            # 4. Store transition for training
            store_transition(observation, action, reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 120:
                done = True

            if done:
                episode_rewards_sum = sum(episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)
                print("==========================================")
                print("Episode: ", episode)
                if episode > upper_limit:
                    print("Seconds: ", elapsed_sec)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)

                # learn
                nabla_w, nabla_b = learn(observation, weights, biases)
                temp_b = biases
                for index, (w, nw) in enumerate(zip(weights, nabla_w)):
                    weights[index] = w + nw
                for index, (b, nb) in enumerate(zip(biases, nabla_b)):
                    biases[index] = b + nb

                ##########
                episode_observations = []
                episode_rewards = []
                episode_actions = []
                break
            observation = observation_

    # choose action

    # save state|actions|state

    # if done -> learn

