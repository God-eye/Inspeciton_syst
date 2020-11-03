# Intelligent Inspection System
---
## About
Present surveillance systems are generally used as a passive source of information which doesn't help in preventing any abnormal activities which can lead to serious problems and even cost human lives. If we could convert it into an active source of information then we can prevent those kinds of loss. There are several solutions available for this problem one is to hire a human who will be continuously monitoring the CCTV feed and take relevant actions when needed, the second is if we use hardware solutions like motion detectors and fire sensors, etc. which will detect abnormal activities and inform the owner. However, these solutions have their limits firstly they are quite expensive and if you are using hardware-based solutions then you will have to install a large no of sensors all around the property you want to monitor and they will have a limited no of abnormal activities they can identify like fire or break-in, etc. But if we use AI-based software solutions then, first of all, it can detect a large no of such abnormal activities and as it has its own intelligence it can automatically take relevant actions in order to prevent serious situations. And it's significantly cheaper than any other solution available in the market.

---

## Files

* #### Fronend 
Contains all the files related to the web app

* #### Test
Contains the demo test videos

* #### Train
Contains all the scripts used for training

* #### model_weights
contains the weights of Anomaly detection model

* #### result
 contains the output text file

* #### model_v2.py 
   Model test file

---

## Requirement
This is a little computationally demanding software. For it to run in Real-Time we would suggest a system with the following specifications.
* i3 6th gen+
* 8 GB Ram
* 5 GB V-Ram
* Nvidia GTX 1050 or later

## Build





---

## Challenges we ran into
we faced difficulties to reduce the false positives of the system to prevent irrelevant alarms. we modified the architecture to include a small algorithm to detect false positives and prevent the system from activating the alarms when not required.
