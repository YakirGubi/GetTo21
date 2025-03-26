import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
import main
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(main.NUMBER_OF_CARDS, 256, main.NUMBER_OF_CARDS*2).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self):
        return main.CARDS

    def remember(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_memory = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_memory = self.memory

        states, actions, rewards, dones = zip(*mini_memory)
        self.trainer.train_step(states, actions, rewards, dones)

    def train_short_memory(self, state, action, reward, done):
        self.trainer.train_step(state, action, reward, done)

    def get_action(self, state):
        self.epsilon = 1500 - self.n_games
        final_move = np.zeros(20)

        if random.randint(0, 2000) < self.epsilon:
            final_move[random.randint(0, 9)] = 1
            final_move[random.randint(10, 19)] = 1

        else:
            state_now = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state_now)
            final_move[torch.argmax(prediction[:10])] = 1
            final_move[torch.argmax(prediction[10:]) + 10] = 1

        return final_move

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    total_wins = 0
    agent = Agent()

    while True:
        state = agent.get_state()
        action = agent.get_action(state)
        reward, score, done = main.action(action)

        agent.train_short_memory(state, action, reward, done)
        agent.remember(state, action, reward, done)

        main.restart_game()
        agent.n_games += 1
        agent.train_long_memory()

        plot_score.append(score)
        total_score += score
        mean_score = np.mean(plot_score)
        plot_mean_score.append(mean_score)

        if score == main.GOAL:
            total_wins += 1

        if agent.n_games % 100 == 0:
            plot(plot_score, plot_mean_score)

            print('Wins: ', total_wins/agent.n_games *100, '%')


if __name__ == '__main__':
    train()