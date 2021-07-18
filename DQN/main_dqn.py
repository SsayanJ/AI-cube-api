import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    n_games = 500
    best_score = -np.inf
    agent = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8,
                     n_actions=4, mem_size=1_000_000, epsilon_end=0.01,
                     batch_size=64)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[max(0, i-100): i+1])

        print('episode: ', i, 'score %.2f: ' % score,
              ' average score %.2f' % avg_score)

        if avg_score > best_score:
            agent.save_model()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    filename = 'lunarlander.png'
    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, eps_history, filename)
