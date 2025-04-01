import torch
import random
import numpy as np

GOAL = 21
NUMBER_OF_CARDS = 10
CARDS = np.random.randint(1, GOAL, size=NUMBER_OF_CARDS)

def restart_game():
    CARDS = np.random.randint(1, GOAL, size=NUMBER_OF_CARDS)

    must = np.random.choice(np.arange(NUMBER_OF_CARDS), size=2, replace=False)
    CARDS[must[0]] = random.randint(1, GOAL)
    CARDS[must[1]] = GOAL - CARDS[must[0]]

def action(action):
    temp_action = torch.tensor(action, dtype=torch.float)
    card1 = torch.argmax(temp_action[:10]).item()
    card2 = torch.argmax(temp_action[10:]).item()

    print('Number:', CARDS[card1] + CARDS[card2])

    done = True
    score = GOAL - abs(GOAL - (CARDS[card1] + CARDS[card2]))

    if card1 == card2:
        reward = -50
        score = 0

    elif CARDS[card1] + CARDS[card2] == GOAL:
        reward = 50

    else:
        reward = -abs(GOAL - (CARDS[card1] + CARDS[card2]) + 10)

    return reward, score, done