import logging
logging.basicConfig(filename = 'U_P_LSTM_Transfer_TO_SJ.txt', filemode = 'w', level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.ssaqn_agent import SSAQNAgent
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode
from pyfiction.simulators.games.catsimulator2016_simulator import CatSimulator2016Simulator
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator
from pyfiction.simulators.games.starcourt_simulator import StarCourtSimulator
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator
from pyfiction.simulators.games.transit_simulator import TransitSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

"""
An SSAQN agent that supports for testing transfer learning on pre-trained models
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='file path of a model to load',
                    type=str)

args = parser.parse_args()
model_path = args.model
print("Path given is ", model_path)
#ch = input("Proceed?")
simulator = TheRedHairSimulator()
test_steps = 1

print('Training on game:', simulator.game.name)
print('Testing on game:', simulator.game.name)

agent = SSAQNAgent(train_simulators=simulator)

# Load or learn the vocabulary (random sampling on many games could be extremely slow)
agent.initialize_tokens('vocabulary.txt')

optimizer = RMSprop(lr=0.000001)

embedding_dimensions = 300
lstm_dimensions = 100
dense_dimensions = 50

agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer, 
                   embeddings = '../../../../glove.840B.300d.txt', 
                   embeddings_trainable = False)

agent.model = load_model(model_path)
#ch = input("Load complete, begin train?")
# Visualize the model
try:
    plot_model(agent.model, to_file='model.png', show_shapes=True)
except ImportError as e:
    logger.warning("Couldn't print the model image: {}".format(e))

# Transfer learning test - train the agent on the previously unseen (only used for testing) game

agent.train_online(episodes=256, batch_size=64, gamma=0.95, epsilon=1, epsilon_decay=0.99,
                   prioritized_fraction=0.25, test_interval=1, test_steps=test_steps,
                   log_prefix=('transfer' + str(1)))
