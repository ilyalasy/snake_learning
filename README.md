# Snake Learning
Small project to beat the snake game with use of deep reinforcement learning and computer vision.
Used techs and libraries:
* [OpenCV](https://github.com/skvark/opencv-python) for object recognition
* [Tensorforce](https://github.com/tensorforce/tensorforce) for Deep RL (DQN in particular)

## Implementation ideas 
* Treat every 4 frames (for motion detection) of the game as a **state** for DQN.
* Use CNN as spec for DQN to convolute frames.
* DQN then generates **action** (in our case *left*, *right*, *up*, *down* keys).
* Now the tricky part: figure out the **reward**.
  * Snake specific approach: based on two states (before and after action) figure out whether agent moved closer to apple.
  * More generalized: create model for reward and next frame ***prediction***. [Baseline article](https://arxiv.org/abs/1611.07078)
* Feed the **reward** to DQN.
* TRAIN TILL DEATH!

## P.S.
Right now all CV is made based on playsnake.org, but there are more online snake games to support in future.
