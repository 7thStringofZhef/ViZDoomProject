# Enormous parameter class
from vizdoom import GameVariable

class doomParams(object):
    scenarioPath='CS7180Scenario.cfg'
    numFrames=2000000
    framesBetweenEvaluations=50000

    #RAINBOW COMPONENTS
    prioritizedReplay = 1
    noisyLinear = 1
    dueling = 1
    double = 1
    multiStep = 3
    distributed = 1

    #[NET PARAMS]
    recurrent=1
    startEps=1.0
    endEps=0.1
    epsSteps=100000
    gamma=0.99
    batchSize=32
    frameskip=4
    framesBeforeTraining=80000
    learningRate=0.001
    recurrenceHistory=4
    trainingFrequency=4
    numActions=3

    #[RAINBOW PARAMS]
    targetUpdateFrequency=1000
    noisyParam=0.5
    priorityBetaStart=0.4
    priorityBetaEnd=1
    priorityBetaSteps=1000000
    priorityOmega=0.5
    atoms = 51
    vMin = -10
    vMax = 10

    #Other
    replayMemoryCapacity=250000
    inputShape = (3, 60, 108)  # Channels*height*width
    hiddenDimensions = 512
    gameVariables=[GameVariable.AMMO2, GameVariable.HEALTH]
    gameVariableBucketSizes=[1, 1]
    numGameVariables=2

    def __init__(self, prioritizedReplay=1, noisyLinear=1, dueling=1, double=1, multiStep=3):
        self.prioritizedReplay = prioritizedReplay
        self.noisyLinear = noisyLinear
        self.dueling = dueling
        self.double = double
        self.multiStep = multiStep


