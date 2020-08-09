---
layout: page
title: Cloud
---
URL: https://www.matpool.com
price: 3RMB/hour
GPU: GeForce RTX 2080 Ti

```
import torch
torch.cuda.current_device()
tourch.cuda.device_count()
tourch.cuda.get_device_name(0)
```

Fire Transfer

```
ssh -p <port> root@hz-t2.matpool.com
scp -r -P $PORT ./LocaltoCloud root@hz-t2.matpool.com:
scp -r -P $PORT root@hz-t2.matpool.com:./CloudtoLocal /home/blp/Desktop/
```

Change the mirror file

```
scp -P $PORT /etc/apt/sources.list root@hz-t2.matpool.com:/etc/apt/sources.list
```

