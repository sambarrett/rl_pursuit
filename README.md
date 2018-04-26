# Building
This project is configured using a Makefile, so you should be able to build it through: `make`.

The build process has only been tested on Ubuntu.  It requires a number of packages.  The commands below are expected to pull in all dependencies, many of which are not explicitly listed.
Basics:
```
apt-get install make build-essential g++ git
```
C++ library dependencies:
```
apt-get install libarmadillo-dev libboost-dev libboost-thread-dev libgflags-dev libboost-python-dev python2.7-dev
```
Install Java dependencies for weka bridge:
```
apt-get install openjdk-8-jdk-headless
```

# Running
The `main` executable takes in json config files to decide what to run.  Some basic examples are included in `configs/`.  An example of how to run a trial is:
```
bin/64/main configs/all_greedy.json
```

To run a trial with an ad hoc agent running a MCTS policy, knowing that its teammates are greedy:
```
bin/64/main configs/greedy.json configs/greedy_planner.json
```

To run a trial with an ad hoc agent running a MCTS policy, selecting from 4 types of teammates using Bayesian updates to its expectations:
```
bin/64/main configs/greedy.json configs/planner_bayesian.json
```

To run a trial with an ad hoc agent running a MCTS policy, selecting from 4 types of teammates using polynomial weights updates to its expectations:
```
bin/64/main configs/greedy.json configs/planner_poly.json
```


Other types of teammates include:
- gr
- ta
- gp
- pd

See [src/factory/AgentFactory.cpp](src/factory/AgentFactory.cpp) for more agent types.
