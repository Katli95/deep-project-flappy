from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
import random
import numpy as np
import pickle
from pprint import pprint
from timeit import default_timer as timer
import os
import csv

from collections import deque
import cv2

from keras.initializers import normal as normal_initializer
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import adam
import tensorflow as tf

reward_structures = {
    "basicFlappyAgent": {"positive": 1.0, "tick": 0.0, "loss": -5.0},
    "improvedFlappyAgent": {"positive": 1.0, "tick": 0.1, "loss": -1.0},
    "ActualGame": {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
}

class FlappyDeepQAgent:
    def __init__(self, reward_values=reward_structures["improvedFlappyAgent"], learning_rate=1e-5, discount_factor = 0.95, initial_epsilon=1, epsilon_decay_rate=0.9998, final_epsilon=0.1, mini_epochs=7, training_episodes=20000, batch_size=32, reload_model=False, reload_weights=False, updates_to_network=0, model_type="", epsilon_decay_type="sinusoidal"):
        self.rewards = reward_values
        self.learningRate = learning_rate
        self.discountFactor = discount_factor
        self.batchSize = batch_size
        self.epsilon = initial_epsilon
        self.initialEpsilon = initial_epsilon
        self. epsilonDecayType = epsilon_decay_type
        if epsilon_decay_type == "sinusoidal":
            self.epsilonDecayRate = epsilon_decay_rate
        else:
            self.final_epsilon = final_epsilon
        self.miniEpochs = mini_epochs
        self.trainingEpisodes = training_episodes
        self.replayMem = replayMemory(30000)
        self.observationThreshold = 3000
        self.updatesToNetwork = updates_to_network
        self.printCnt = 500
        self.modelType=model_type
        if(reload_model):
            self.QNetwork = load_model("RoutineSave-{}flappyBirdQNetworkModel.h5".format(model_type))
            self.TargetQNetwork = load_model("RoutineSave-{}flappyBirdQNetworkModel.h5".format(model_type))
        else:
            if model_type == "":
                self.QNetwork = getQNetwork(learning_rate)
                self.TargetQNetwork = getQNetwork(learning_rate)
                if reload_weights:
                    self.QNetwork.load_weights("flappyBirdQNetworkWeights.h5")
                    self.TargetQNetwork.load_weights("flappyBirdQNetworkWeights.h5")
            elif model_type == "Representational":
                self.QNetwork = getRepresentationalQNetwork()
                self.TargetQNetwork = getRepresentationalQNetwork()
                if reload_weights:
                    self.QNetwork.load_weights("flappyBirdRepresentationalQNetworkWeights.h5")
                    self.TargetQNetwork.load_weights("flappyBirdRepresentationalQNetworkWeights.h5")



    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return self.rewards

    def observe(self, s1, a, r, s2, isEnd):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        self.replayMem.put((s1, a, r, s2, isEnd))

        if self.isTraining():
            batch = self.replayMem.getBatch(self.batchSize)
            init_states, actions, rewards, next_states, isFinal = list(zip(*batch))
            # if self.modelType == "":
            init_states = np.concatenate(init_states)
            next_states = np.concatenate(next_states)
            # else:
            #     init_states = np.array(init_states)
            #     next_states = np.array(next_states)
            targets = self.TargetQNetwork.predict(init_states)
            estimated_values = self.TargetQNetwork.predict(next_states)
            
            targets[range(self.batchSize), actions] = rewards + (self.discountFactor*np.max(estimated_values, axis=1)*np.invert(isFinal))
            loss = self.QNetwork.train_on_batch(init_states, targets)
            
            self.log("\tLoss: {}".format(loss))

            self.updatesToNetwork += 1
            self.updateEpsilon()

            if self.updatesToNetwork % 1000 == 0:
                self.saveModel("RoutineSave")
                self.updateTarget()
                print("Q Network updated {} times".format(self.updatesToNetwork))

    def isTraining(self):
        return self.replayMem.size() > self.observationThreshold

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        if random.random() < self.epsilon:
            retval = random.randint(0,1)
        else:
            q = self.QNetwork.predict(state)
            self.log("Exploiting!\m{}".format(*q))
            self.printCnt -= 1
            retval = np.argmax(q)
        return retval


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        q = self.QNetwork.predict(state)
        return np.argmax(q)

    def saveModel(self, prefix):
        self.QNetwork.save("{}-{}flappyBirdQNetworkModel.h5".format(prefix, self.modelType))

    def updateTarget(self):
        self.QNetwork.save_weights("{}flappyBirdQNetworkWeights.h5".format(self.modelType), overwrite=True)
        self.TargetQNetwork.load_weights("{}flappyBirdQNetworkWeights.h5".format(self.modelType))

    def updateEpsilon(self):
        newVal = self.epsilon
        if self.epsilonDecayType == "sinusoidal":
            newVal = self.initialEpsilon*(self.epsilonDecayRate**self.updatesToNetwork)*(1/2)*(1+np.cos((2*np.pi*(self.updatesToNetwork%self.trainingEpisodes)*self.miniEpochs)/self.trainingEpisodes))
        elif self.epsilon > self.final_epsilon:
            newVal = np.max([self.final_epsilon, self.epsilon - (self.initialEpsilon-self.final_epsilon)/20000])
        elif self.updatesToNetwork < 100000: 
            return
        else:
             self.epsilon = 0
             return
        self.epsilon = newVal
        self.log("Epsilon: {}".format(self.epsilon))

    def log(self, msg):
        if(self.printCnt < 20):
            print(msg)
            if(self.printCnt < 0):
                self.printCnt = 500

class replayMemory:
    def __init__(self, size):
        self.terminals = deque()
        self.pipePasses = deque()
        self.ticks = deque()
        self.maxSize = size//3

    def size(self):
        return len(self.terminals)+len(self.pipePasses)+len(self.ticks)

    def put(self, observation):
        reward = observation[2]
        if reward < 0:
            self.enqeue(self.terminals, observation)
        elif reward > 0.5:
            self.enqeue(self.pipePasses, observation)
        else:
            self.enqeue(self.ticks, observation)

    def enqeue(self, qeue, observation):
        qeue.append(observation)
        if len(qeue) > self.maxSize:
            qeue.popleft()
        

    def getBatch(self, batch_size):
        numItems = batch_size//3

        passes = min(len(self.pipePasses),numItems)
        deaths = min(len(self.terminals),numItems)
        ticks = min(len(self.ticks),numItems)

        while sum([passes, deaths, ticks]) < batch_size:
            if len(self.pipePasses) > passes:
                passes +=1
            elif len(self.terminals) > deaths:
                deaths +=1
            elif len(self.ticks) > ticks:
                ticks +=1
            else:
                break
        all_in_batch = []
        if deaths > 0:
            all_in_batch.append(random.sample(self.terminals, deaths))
        if passes > 0:
            all_in_batch.append(random.sample(self.pipePasses, passes))
        if ticks > 0:
            all_in_batch.append(random.sample(self.ticks, ticks))
        return np.concatenate(all_in_batch)
        
            
        

img_rows , img_cols = 84, 84
img_channels = 4

def getQNetwork(LEARNING_RATE):
    initializer = normal_initializer(0, 0.01, 42)
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=4, padding='same',input_shape=(img_rows,img_cols,img_channels), kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4,4), strides=2, padding='same', kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3,3), strides=1, padding='same', kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(Dense(2, kernel_initializer=initializer))
    model.add(Activation("linear"))

    opt_adam = adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=opt_adam)
    return model

def getRepresentationalQNetwork():
    model = Sequential()
    model.add(Dense(15, input_shape=(5,), kernel_initializer="normal"))
    model.add(Activation("relu"))
    model.add(Dense(10, kernel_initializer="normal"))
    model.add(Activation("relu"))
    model.add(Dense(2, kernel_initializer="normal"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model

def parseStateRepresentation(rawState):
    # print(rawState)
    x_diff = rawState["next_pipe_dist_to_player"]
    next_pipe_bot = rawState["next_pipe_bottom_y"]
    next_pipe_top = rawState["next_pipe_top_y"]
    y_diff_top = next_pipe_top - rawState["player_y"]
    y_diff_bot = next_pipe_bot- rawState["player_y"]
    player_vel = rawState["player_vel"]
    isUnderPipe = x_diff < 26
    return np.array([[x_diff, y_diff_bot, y_diff_top, player_vel, float(isUnderPipe)]])

def processImage(rawImg):
    # show_image(img)
    img = rawImg[0:288, 0:412]
    # img = np.zeros((img_rows, img_cols))
    img = cv2.resize(img, (img_rows,img_cols))
    img = cv2.flip(img, 0)
    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    img = cv2.warpAffine(img,M,(cols,rows))
    # show_image(img)
    # show_image(img)
    normImg = img / 255
    # cv2.normalize(img, normImg, 0, 1)
    # _, normImg = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    # show_image(normImg)
    return normImg

def showState(state):
    sperate_images = [state[0,:,:,3],state[0,:,:,2],state[0,:,:,1],state[0,:,:,0]]
    numpy_horizontal_concat = np.concatenate(sperate_images, axis=1)

    cv2.imshow('Numpy Vertical Concat', numpy_horizontal_concat)

    cv2.waitKey()

def show_image(img):
    cv2.imshow("Debug Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bestAverage = 0

def run_game(agent, train, teaching_agent=None):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    if train:
        reward_values = agent.reward_values()
        dispScreen = False
        force_fps = True
    else:
        reward_values = reward_structures["ActualGame"]
        dispScreen = True
        force_fps = False

    env = PLE(FlappyBird(), fps=30, display_screen=dispScreen, force_fps=force_fps, rng=None,
              reward_values=reward_values)

    env.init()

    current_state_representation = env.game.getGameState()

    current_state = None
    if agent.modelType == "Representational":
        current_state = parseStateRepresentation(current_state_representation)
    else: 
        current_state = constructStateFromSingleFrame(processImage(env.getScreenGrayscale()))

    score = 0
    scores = {}
    
    startTime = timer()
    frames = 0
    episodes = 0
    while True:
        frames += 1
        # pick an action
        if teaching_agent is not None:
            action = teaching_agent.policy(current_state_representation)
        else: 
            if train:
                action = agent.training_policy(current_state)    
            else:
                action = agent.policy(current_state)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        if(reward > 0.5):
            score += 1

        current_state_representation = env.game.getGameState()

        next_state = None
        if agent.modelType == "Representational":
            next_state = parseStateRepresentation(current_state_representation)
        else: 
            next_frame = processImage(env.getScreenGrayscale())
            next_frame = next_frame.reshape(1,next_frame.shape[0], next_frame.shape[1], 1)
            #Append the new frame to the front of the current state representation to construct the new state
            next_state = np.append(next_frame, current_state[:,:,:,:3], axis=3)


        # showState(next_state)
        if train:
            agent.observe(current_state, action, reward,
                        next_state, env.game_over())

        current_state = next_state
        
        # reset the environment if the game is over
        if env.game_over():
            episodes +=1 
            if agent.updatesToNetwork > 0:
                if score not in scores:
                    scores[score] = 0
                scores[score] += 1
                printScores(scores, frames)

                currentAverage = logScore(scores, agent.updatesToNetwork)
                if currentAverage > bestAverage + 0.2:
                    agent.saveModel("BestSoFar")
            score = 0
            
            env.reset_game()
            current_state_representation = env.game.getGameState()

            if agent.modelType == "Representational":
                current_state = parseStateRepresentation(current_state_representation)
            else: 
                current_state = constructStateFromSingleFrame(processImage(env.getScreenGrayscale()))

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

def logScore(scores, frames):
        with open("scores.csv", "a", newline='') as file:
            writer = csv.writer(file)
            if os.stat("scores.csv").st_size == 0:
                writer.writerow(['Updates_To_Network',"Average_Score"])
            avg = getAverageOfScores(scores)
            writer.writerow([frames,avg])
            return avg
        
def getAverageOfScores(scores):
    sumOfScores = 0
    totalEntries = 0
    for score, frequency in scores.items():
        totalEntries += frequency
        sumOfScores += score*frequency
    return sumOfScores / totalEntries
