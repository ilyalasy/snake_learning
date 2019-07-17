
from tensorforce.execution import Runner
from environment import SnakeEnvironment 
import dqn
import logger

MAX_EPISODES = 1000
MAX_TIMESTEPS = 1000

def main():
    log = logger.get_logger()
    env = SnakeEnvironment()
    agent = dqn.get_agent()

    runner = Runner(agent, env)
        
    report_episodes = 10

    def episode_finished(r):       
        if r.episode % report_episodes == 0:
            log.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            log.info("Episode reward: {}".format(r.episode_rewards[-1]))
            log.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    log.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(MAX_EPISODES, MAX_TIMESTEPS, episode_finished=episode_finished)
    runner.close()
    agent.save_model("/models")
    log.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()