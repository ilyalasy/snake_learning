from tensorforce.agents import DQNAgent
from environment import STATES, ACTIONS

LEARNING_RATE = 0.0002
EPSILON = 1e-5


def get_agent():

    exploration = dict(type='epsilon_decay',
                       final_epsilon=0.01, timesteps=1000)
    optimizer = dict(type='rmsprop', learning_rate=LEARNING_RATE)
    network_spec = [
        dict(type='conv2d', size=32, activation='elu',
             window=[8, 8], stride=(4, 4), padding="VALID"),
        dict(type='tf_layer', layer='batch_normalization', epsilon=EPSILON),
        dict(type='conv2d', size=64, activation='elu',
             window=[4, 4], stride=(2, 2), padding="VALID"),
        dict(type='tf_layer', layer='batch_normalization', epsilon=EPSILON),
        dict(type='conv2d', size=128, activation='elu',
             window=[4, 4], stride=(2, 2), padding="VALID"),
        dict(type='tf_layer', layer='batch_normalization', epsilon=EPSILON),
        dict(type='flatten'),
        dict(type='dense', size=512, activation='elu')
    ]

    agent = DQNAgent(
        states=STATES,
        actions=ACTIONS,
        network=network_spec,
        optimizer=optimizer,
        discount=0.95,
        actions_exploration=exploration
    )
    return agent
