from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
import random
import numpy as np
import pickle
from pprint import pprint
from timeit import default_timer as timer
from abc import ABC, abstractmethod

from collections import deque
import cv2 #import resize, threshold, THRESH_BINARY, normalize, NORM_MINMAX, imshow

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , RMSprop
import tensorflow as tf

reward_structures = {
    "basicFlappyAgent": {"positive": 1.0, "tick": 0.0, "loss": -5.0},
    "improvedFlappyAgent": {"positive": 1.0, "tick": 0.1, "loss": -1.0},
    "game": {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
}


class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return reward_structures["improvedFlappyAgent"]

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1)

class FlappyDeepQAgent(FlappyAgent):
    def __init__(self, reward_values=reward_structures["improvedFlappyAgent"], learning_rate=1e-6, discount_factor = 0.95, initial_epsilon=1, final_epsilon = 0.1, batch_size=32):
        self.rewards = reward_values
        self.learningRate = learning_rate
        self.discountFactor = discount_factor
        self.finalEpsilon = final_epsilon
        self.batchSize = batch_size
        self.epsilon = initial_epsilon
        self.QNetwork = getQNetwork(learning_rate)
        self.TargetQNetwork = getQNetwork(learning_rate)
        self.replayMem = deque()
        self.observationThreshold = 3000
        self.replayMemMaxSize = 20000
        self.updatesToNetwork = 0

    def reward_values(self):
        return self.rewards

    def observe(self, s1, a, r, s2, isEnd):
        # first_state_index = improved_map_state(s1)
        # second_state_index = improved_map_state(s2)
        # if end:
        #     self.q_table[second_state_index] = [r, r]
        # old_value = self.q_table[first_state_index][a]
        # self.q_table[first_state_index][a] = (1-self.learning_rate)*old_value + self.learning_rate*(
        #     r + self.discount_factor*max(self.q_table[second_state_index]))
        self.replayMem.append((s1, a, r, s2, isEnd))
        if len(self.replayMem) > self.observationThreshold:
            batch = random.sample(self.replayMem, self.batchSize)

            init_states, actions, rewards, next_states, isFinal = zip(*batch)
            init_states = np.concatenate(init_states)
            next_states = np.concatenate(next_states)
            targets = self.TargetQNetwork.predict(init_states)
            estimated_values = self.TargetQNetwork.predict(next_states)
            
            targets[range(self.batchSize), actions] = rewards + (self.discountFactor*np.max(estimated_values, axis=1)*np.invert(isFinal))
            loss = self.QNetwork.train_on_batch(init_states, targets)

            self.updatesToNetwork += 1

            if self.updatesToNetwork == 1000:
                self.saveModel()
                self.updateTarget()
                self.updatesToNetwork = 0


    def training_policy(self, state):
        # print("state: %s" % state)
        # p = random.random()
        # if p < self.epsilon:
        #     return random.randint(0, 1)
        # else:
        #     stateIndex = improved_map_state(state)
        #     return np.argmax(self.q_table[stateIndex])
        if random.random() < self.epsilon:
            return random.randint(0,1)
        else:
            q = self.TargetQNetwork.predict(state)
            return np.argmax(q)

    def policy(self, state):
        # print("state: %s" % state)
        # stateIndex = improved_map_state(state)
        # return np.argmax(self.q_table[stateIndex])
        q = self.TargetQNetwork.predict(state)
        return np.argmax(q)

    def saveModel(self):
        self.QNetwork.save_weights("flappyBirdQNetwork.h5", overwrite=True)
        with open("flappyBirdQNetwork.json", "w") as outfile:
            json.dump(self.QNetwork.to_json(), outfile)

    def updateTarget(self):
        self.TargetQNetwork.load_weights("flappyBirdQNetwork.h5")


img_rows , img_cols = 84, 84
img_channels = 4

def getQNetwork(LEARNING_RATE = 1e-6):
    model = Sequential()
    model.add(Convolution2D(32, 8, strides=4, padding='same',input_shape=(img_rows,img_cols,img_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    rmsprop = RMSprop(lr=LEARNING_RATE, decay=0.9, rho=0.95)
    model.compile(loss='mse',optimizer=rmsprop)
    print("We finish building the model")
    return model

def processImage(img):
    # show_image(img)
    img = img[0:288, 0:412]
    # show_image(img)
    img = cv2.resize(img, (img_rows,img_cols))
    # show_image(img)
    # normImg = np.zeros((img_rows, img_cols))
    # cv2.normalize(img, normImg, 0, 1, cv2.NORM_MINMAX)
    _, normImg = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    # show_image(normImg)
    return normImg

def show_image(img):
    cv2.imshow("Debug Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_game(nb_episodes, agent, train):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    if train:
        reward_values = agent.reward_values()
        dispScreen = False
        force_fps = True
    else:
        reward_values = reward_structures["game"]
        dispScreen = True
        force_fps = False

    env = PLE(FlappyBird(), fps=30, display_screen=dispScreen, force_fps=force_fps, rng=None,
              reward_values=reward_values)

    env.init()
    # current_state = env.game.getGameState()
    current_state = constructStateFromSingleFrame(processImage(env.getScreenGrayscale()))

    score = 0
    scores = {}
    startTime = timer()
    frames = 0
    while nb_episodes > 0:
        frames += 1
        # pick an action
        if train:
            action = agent.training_policy(current_state)
        else:
            action = agent.policy(current_state)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        if reward == 1:
            score += reward

        # TODO: for training let the agent observe the current state transition
        next_frame = processImage(env.getScreenGrayscale())
        next_frame = next_frame.reshape(1, next_frame.shape[0], next_frame.shape[1], 1)
        #Append the new frame to the front of the current state representation to construct the new state
        next_state = np.append(next_frame, current_state[:,:,:,:3], axis=3)
        agent.observe(current_state, action, reward,
                      next_state, env.game_over())
        current_state = next_state
        
        # reset the environment if the game is over
        if env.game_over():
            if score not in scores:
                scores[score] = 0
            scores[score] += 1
            if nb_episodes % 500 == 0:
                print(action)
                printScores(scores, frames)
            env.reset_game()
            current_state = constructStateFromSingleFrame(processImage(env.getScreenGrayscale()))
            nb_episodes -= 1
            score = 0
        # Safety break
        if( frames > 1000000):
            break
        if frames % 1000 == 0:
            agent.saveModel()
    pygame.display.quit()
    printScores(scores, frames)
    print((timer() - startTime) / 60, " minutes")

def constructStateFromSingleFrame(frame):
    tempState = np.stack((frame, frame, frame, frame),axis=2)
    return tempState.reshape(1, tempState.shape[0], tempState.shape[1], tempState.shape[2])

def printScores(scores, frames):
    total = sum(scores.values())
    print("scores after " + str(total) + " iteration(s)")
    percentage = {}
    for key, val in scores.items():
        percentage[key] = "{:.2f}".format(
            val/total*100) + "% : (" + str(val) + ")"
    pprint(percentage)
    print("Total frames : ", frames)