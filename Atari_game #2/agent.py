import torch 
import pygame
import numpy as np 
from spaceship import SpaceShipAI 
from collections import deque, namedtuple
import random 
from model import Linear_QNet, QTrainer

# constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
MAX_MEMORY = 100_000
LR = 0.001
BATCH_SIZE = 1000



class AgentAI:

    def __init__(self):
        self.n_games = 0 
        self.epsilon = 0 # controls randomness 
        self.gamma  = 0.9 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(34, 128, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def get_state(self, game):

        # state is a d2 array that the first elemnt is player_x, player_y, block1_state, block2_state, ... block12_state 
        state = [0 for i in range(0, 34)]

        state[0] = game.player_rect.y

        counter = 1
        for i in range(0, 3):
            for j in range(0, 11):
                if [i*40 + 80, j*40] in game.obstacles:
                    state[counter] = 1

                counter += 1
        return state


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 300 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 300) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1 

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() 
            final_move[move] = 1 

        return final_move


def train():

    pygame.init()

    game = SpaceShipAI(window_size)

    agent = AgentAI()

    while True: 

        # pygame.time.delay(5)

        # get current state
        old_state = agent.get_state(game)

        # get action 
        action = agent.get_action(old_state)

        # update game and get info based on action
        reward, score, done = game.play_step(key_pressed=np.argmax(np.array(action)), previous_score=game.score)

        # get new state
        new_state = agent.get_state(game)

        # train short-term memory according to info 
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # remember stuff 
        agent.remember(old_state, action, reward, new_state, done)

        # check if game over (if game over, update the target model)
        if done:

            agent.n_games += 1 
            agent.train_long_memory()

            if game.score > game.highscore:
                game.highscore = game.score
                agent.model.save()
            
            print("Game:", agent.n_games, "Score:", game.score, "Record:", game.highscore)

            game.reset() 

if __name__ == "__main__":

    train()