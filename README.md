# deep-project-flappy

## Preparing the invironment
For the model to work properly one must modify pygame's version of flappy bird slightly. This was done to simplify the background, control the colors in the game and fix it's state representation slightly.

This can be done by copying the __init__moidified.py file and replace pygames default \_\_init\_\_.py and add the background-empty.png images to its assets.

## Running the code
*Be advised, running the code will result in the creation of a scores.csv file and training will save the model in a myriad of ways*

To test, run:
```python
from flappy_agent import *
agent = FlappyDeepQAgent(reload_model=True, reload_path="120-flappyBirdQNetworkModel.h5", model_type="")
run_game(agent, False)
```

To train, run:
```python
from flappy_agent import *
teacher = FlappyDeepQAgent(reload_model=True, reload_path="6x18-12-8-improved-BestSoFar-RepresentationalflappyBirdQNetworkModel.h5", model_type="Representational")
agent = FlappyDeepQAgent()
run_game(agent, train=True, teaching_agent=teacher)
```