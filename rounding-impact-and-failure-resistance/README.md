# Failure Resistance and Rounding Impact

In this folder are presented the source code confirming the negligible impact of the rounding
as well as the failure resistance property featured by our protocols.

## Failure Resistance

The failure resistance in the secure federated multi-armed bandits setting allows a protocol
to return almost the same rewards in case where a data owner fails and does not respond during the protocol execution.

By construction, the rewards saving mechanism we have implemented depends only on the correctness of the used cryptographic technique to send the current local cumulative rewards to the federation server. And since the correctness property
is assumed for our primitives, we have decided to write a dedicated approach measuring the potential impact of the 
rewards sending mechanism.

To run the script, execute the following command:
```sh
pip3 install -r requirements.txt && python3 failure-resistance.py
```

## Rounding Impact on Discrete Bandits

Rounding may reduce the performance of the plugged multi-armed bandits algorithm. To measure precisely this impact,
we have written a Python script whose the goal is to measure the returned total cumulative rewards with and without score rounding. 

To run the script, execute the following command:
```sh
pip3 install -r requirements.txt && python3 discrete-bandits.py
```