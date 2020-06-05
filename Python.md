---
layout: page
title: Python
---

1. enumerate:

   ```python
   a=    [[1.        , 0.12390437],
          [2.        , 0.15133714],
          [3.        , 0.1848436 ],
          [2.        , 0.53991488]]
        
   for s_prime,(reward,probability) in enumerate a:
       print(s_prime)
       print(reward)
       print(probability)
       
   Result:
   0
   1.0
   0.12390437
   1
   2.0
   0.15133714
   2
   3.0
   0.1848436
   3
   2.0
   0.53991488
   ```

2. argwhere: np.argwhere(k==np.max(k))  this is used because np.argmax only return one index even if there are multiple same values

   