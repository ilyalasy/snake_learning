import numpy as np
import time
import logging

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner

from environment import SnakeEnvironment

logging.basicConfig(level=logging.INFO)

def main():
    env = SnakeEnvironment()

    network_spec = [
        dict(type='conv2d', size=32, activation='elu', window=[8,8], stride=(4,4),padding = "VALID"),
        dict(type='tf_layer', layer='batch_normalization'),
        dict(type='conv2d', size=64, activation='elu', window=[4,4], stride=(2,2),padding = "VALID"),
        dict(type='tf_layer', layer='batch_normalization'),
        dict(type='conv2d', size=128, activation='elu', window=[4,4], stride=(2,2),padding = "VALID"),
        dict(type='tf_layer', layer='batch_normalization'),
        dict(type='flatten'),
        dict(type='dense', size=None)
    ]

    agent = DQNAgent(
        states=env.states,
        actions=env.actions,
        network=network_spec
    )

    max_episodes = 10000
    max_timesteps = 1000
    runner = Runner(agent, env)
        
    report_episodes = 1

    def episode_finished(r):       
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        env.mover.click_center()
        time.sleep(0.25)
        env.mover.start_game()
        time.sleep(0.05)
        env.mover.start_game()
        time.sleep(2)
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()