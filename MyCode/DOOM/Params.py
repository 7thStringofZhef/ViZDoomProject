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

    #[NET PARAMS]
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
    numRecurrentUpdates=2
    embeddingDim=32
    numActions=3
    dropout=0.5

    #[RAINBOW PARAMS]
    targetUpdateFrequency=1000
    noisyParam=0.5
    priorityBetaStart=0.4
    priorityBetaEnd=1
    priorityBetaSteps=200000
    priorityOmega=0.5

    #Other
    replayMemoryCapacity=250000
    inputShape = (3, 60, 108)  # Channels*width*height
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


