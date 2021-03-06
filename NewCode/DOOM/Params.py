# Enormous parameter class
from vizdoom import GameVariable

class doomParams(object):
    scenarioPath='CS7180Scenario.cfg'
    numFrames=500000
    framesBetweenEvaluations=20000
    framesBetweenSaves=20000

    modelName="Test"

    #RAINBOW COMPONENTS
    prioritizedReplay = 1
    noisyLinear = 1
    dueling = 1
    double = 1
    multiStep = 3
    distributed = 1

    #[NET PARAMS]
    recurrent=0
    startEps=1.0
    endEps=0.01
    epsSteps=250000
    gamma=0.99
    batchSize=32
    frameskip=4
    framesBeforeTraining=1000
    learningRate=0.001
    sequenceLength=4
    trainingFrequency=4
    numActions=3

    #[RAINBOW PARAMS]
    targetUpdateFrequency=1000
    noisyParam=0.5
    priorityBetaStart=0.4
    priorityBetaEnd=1
    priorityBetaFrames=1000000
    priorityOmega=0.5
    atoms = 51
    vMin = -10
    vMax = 10

    #Other
    replayMemoryCapacity=500000
    inputShape = (3, 60, 108)  # Channels*height*width
    hiddenDimensions = 512
    gameVariables=[GameVariable.AMMO2, GameVariable.HEALTH]
    gameVariableBucketSizes=[1, 1]
    numGameVariables=2

    def __init__(self, modelName, prioritizedReplay=1, noisyLinear=1, dueling=1, double=1, multiStep=3, distributed=1):
        self.prioritizedReplay = prioritizedReplay
        self.noisyLinear = noisyLinear
        self.dueling = dueling
        self.double = double
        self.multiStep = multiStep
        self.distributed = distributed
        self.modelName = modelName


