
from tensorforce.execution import Runner
from environment import SnakeEnvironment
import dqn
from logger import get_logger

MAX_EPISODES = 5000
MAX_TIMESTEPS = 100000

# https://playsnake.org/ 800x600 at top left
PLAYSNAKE_L = 190
PLAYSNAKE_T = 200
PLAYSNAKE_W = 380
PLAYSNAKE_H = 256
PLAYSNAKE_RESTART = {'action': (480, 376), 'wait_for': 'go'}
PLAYSNAKE_OVER = ["game", "ouer", "over", "best", "score"]
PLAYSNAKE_FIELD = {'top': PLAYSNAKE_T, 'left': PLAYSNAKE_L,
                   'width': PLAYSNAKE_W, 'height': PLAYSNAKE_H}

# http://patorjk.com/games/snake/ 800x600 at top left

def is_jssnake_over(image):
    return (image[200,250] == [0,0,0]).all()

import cv2
def jssnake_preprocess(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_channels = cv2.split(hsv)

    rows = image.shape[0]
    cols = image.shape[1]

    for i in range(rows):
        for j in range(cols):
            h = hsv_channels[0][i][j]

            if h > 90 and h < 130:
                hsv_channels[2][i][j] = 255
            else:
                hsv_channels[2][i][j] = 0
    return cv2.merge(hsv_channels)
    

JSSNAKE_L = 30
JSSNAKE_T = 105
JSSNAKE_W = 740
JSSNAKE_H = 440
JSSNAKE_RESTART = {'action': 'enter'}
JSSNAKE_OVER = is_jssnake_over
JSSNAKE_FIELD = {'top': JSSNAKE_T, 'left': JSSNAKE_L,
                 'width': JSSNAKE_W, 'height': JSSNAKE_H}


def get_environment(name):
    if name == 'jssnake':
        return SnakeEnvironment(JSSNAKE_FIELD, JSSNAKE_OVER, JSSNAKE_RESTART,preprocess=jssnake_preprocess)
    if name == 'playsnake':
        return SnakeEnvironment(PLAYSNAKE_FIELD, PLAYSNAKE_OVER, PLAYSNAKE_RESTART)


def main():
    log = get_logger()

    env = get_environment('jssnake')
    agent = dqn.get_agent()

    runner = Runner(agent, env)

    report_episodes = 5

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            log_episode_info(log, r, 100)
        return True

    log.info("Starting {agent} for Environment '{env}'".format(
        agent=agent, env=env))

    try:
        runner.run(num_episodes=MAX_EPISODES,
                   episode_finished=episode_finished)
    except KeyboardInterrupt:
        log.info("Interrupted!")
    except Exception as e:
        log.error(e)
    finally:
        agent.save_model("./models/")
        runner.close()
        log_episode_info(log, runner, len(runner.episode_rewards))
        log.info("Learning finished.")


def log_episode_info(logger, r, n_rewards):
    logger.info("Finished episode {ep} after {ts} timesteps".format(
        ep=r.episode, ts=r.timestep))
    logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
    logger.info("Average of last {} rewards: {}".format(
        n_rewards, sum(r.episode_rewards[-n_rewards:]) / n_rewards))


if __name__ == '__main__':
    main()
