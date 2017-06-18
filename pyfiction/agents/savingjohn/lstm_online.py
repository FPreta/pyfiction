import datetime
import logging
import os

from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Saving John that uses online learning and prioritized sampling
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=SavingJohnSimulator)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=1024, max_steps=100)

optimizer = RMSprop(lr=0.0005)

embedding_dimensions = 16
lstm_dimensions = 16
dense_dimensions = 4

agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer)

# Visualize the model
try:
    plot_model(agent.model, to_file='model.png', show_shapes=True)
except ImportError as e:
    logger.warning("Couldn't print the model image: {}".format(e))

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to a reward of 19.X (with 19.4 being the optimal reward)
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=1024, max_steps=100, batch_size=256, gamma=0.95, epsilon=1, reward_scale=20,
                                 epsilon_decay=0.99, prioritized_fraction=0.25, test_steps=4)
