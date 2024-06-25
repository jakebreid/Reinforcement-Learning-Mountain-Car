import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

class MountainCarAgent:
    def __init__(self):
        self.pos = -0.5
        self.vel = 0.0
    
    def makeDynMove(self, action):
        self.vel += (action * 0.001) - (0.0025)* np.cos(3 * self.pos)
        self.pos += self.vel
        
        if self.pos <= -1.2: 
            self.pos = -1.2
            self.vel = 0

        if self.pos >= 0.5:
            self.pos = 0.5
            self.vel = 0

        reward = -1.0 if (self.pos < 0.5) else 0.0
        return np.array([self.pos, self.vel]), reward, (self.pos >= 0.5)
    
    def reset(self):
        self.pos = -0.5
        self.vel = 0.0

def EGP(q_values, epsilon):
    if rd.random() < epsilon:
        return np.random.choice(len(q_values))
    return np.argmax(q_values)

class FourierBasis:
    def __init__(self, order, state_dim):
        self.order = order
        self.coeff = np.indices([order + 1] * state_dim).reshape(state_dim, -1).T

    def getFeats(self, state):
        norm_st = (state - np.array([-1.2, -0.1])) / (np.array([1.7, 0.2]))
        return np.cos(np.pi * np.dot(self.coeff, norm_st))

myActions = [0, 1, 2]


def sarsa_lambda(car, basis, numEps, alpha, epsilon, lambd):
    weights = np.zeros((3, basis.coeff.shape[0]))
    count = 0
    plotData = []
    
    while count < numEps:
        car.reset()
        state = np.array([-0.5, 0])
        feats = basis.getFeats(state)
        qVals = [np.dot(weights[a], feats) for a in myActions]
        action = EGP(qVals, epsilon)
        eTrace = np.zeros_like(weights)
        totalR = 0
        isComplete = False
        while not isComplete:
            nextSt, reward, isComplete = car.makeDynMove(action)
            nFeats = basis.getFeats(nextSt)
            nextQVals = [np.dot(weights[a], nFeats) for a in myActions]
            nextAct = EGP(nextQVals, epsilon)
            td_error = reward + (1 - isComplete) * nextQVals[nextAct] - qVals[action]
            eTrace[action] += feats
            weights += alpha * td_error * eTrace
            eTrace *= lambd
            totalR += reward
            state = nextSt
            feats = nFeats 
            qVals = nextQVals
            action = nextAct
        plotData.append(totalR)
        count = count + 1

    return weights, plotData

basis = FourierBasis(3, state_dim=2)
car = MountainCarAgent()

allData = []
countMe2 = 0
while countMe2 < 100:
    car.reset()
    weighty, data = sarsa_lambda(car, basis, numEps=300, alpha=0.0001, epsilon=0.15, lambd=0.85)
    allData.append(data)
    print(f"a trial has completed #{countMe2}")
    countMe2 += 1

averageForEach = []
countMe3 = 0
while countMe3 < 300: 
    countMeIndex = 0
    total = 0
    while countMeIndex < 100: 
        total += allData[countMeIndex][countMe3]
        countMeIndex += 1
    averageForEach.append(total)
    countMe3 += 1

countMe4 = 0
while countMe4 < 300:
    averageForEach[countMe4] /= 100
    countMe4 += 1

leSt = []
grSt = []
countMe3 = 0
while countMe3 < 300: 
    countMeIndex = 0
    total = 0
    while countMeIndex < 100: 
        total += (allData[countMeIndex][countMe3] - averageForEach[countMe3]) ** 2
        countMeIndex += 1
    leSt.append(averageForEach[countMe3] - math.sqrt(total / 100))
    grSt.append(averageForEach[countMe3] + math.sqrt(total / 100))
    countMe3 += 1


plt.plot(averageForEach, label = "average reward")
plt.plot(leSt, label = "lower bound standard error")
plt.plot(grSt, label = "upper bound standard error")
plt.ylabel('Total Reward')
plt.xlabel('Episode #')
plt.title('Learning Curve for Mountain Car (averaged over 100 trials)')
plt.legend()
plt.show()
