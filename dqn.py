import numpy as np
import time

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner

from environment import SnakeEnvironment


def main():
    env = SnakeEnvironment()

    network_spec = [
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ]

    agent = DQNAgent(
        states=env.states,
        actions=env.actions,
        network=network_spec
    )

    max_episodes = 10000
    max_timesteps = 1000
    runner = Runner(agent, env)
        
    report_episodes = 10

    def episode_finished(r):
        env.mover.click_center()
        env.mover.click_worm()
        time.sleep(3)
        if r.episode % report_episodes == 0:
            logging.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logging.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logging.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()