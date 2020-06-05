---
layout: page
title: Research Proposal
---

Applying Reinforcement Learning to Hedging:

**Problem Setup:** only discuss S&P 500 and its options

**State Observation:** [gold price, oil price, emerging market index, etc]

**Choice:** buy and sell of options

**Reward:** return over period of time

```python
class HedgeEnv(gym.env):
    '''
    problem setup:
    hold S&P 500, need to buy options as insurance for that position
    the goal is to achieve maximum protection with the lowest total payment
    
    actions space 5:
    Sell Call(0)
    Buy Call(1)
    Sell Put(2)
    Buy Put(3)
    DO NOTHING(4)
    
    observations 5 (can always add more): 
    Gold Price
    S&P 500 Price
    Emerging Market Price
    LIBOR or other money market indicators
    Oil Price
    
    reward:
    the loss or earning at each decision point
    '''
    
    def __init__(self):
        initiate various values
    def _step(self,action):
        take action
        return observation, reward, done, info
    def _reset(self):
        wipe things clean
    def _render(self):
        NO NEED FOR RENDERING    
```

