
# puddle-world

A simple multi-modal discrete RL environment.

![A PuddleWorld environment](doc/puddle_world_env.png)

## Installation

This package is not distributed on PyPI - you'll have to install from source.

```bash
git clone https://github.com/aaronsnoswell/puddle-world.git
cd puddle-world
pip install -e .
```

## Usage

The canonical PuddleWorld environment is a 5x5 problem.
On the 'dry' reward mode, a random policy achieves a gain of μ=-2.8 σ=1.1 and cumulative
return of μ=-179.2 σ=164.7.
PPO2 from stable_baselines with a 2-layer MLP policy converges after 20k steps, and
achieves a gain of μ=-1.2, σ=0.79 and cumulative return of μ=-10.0 σ=7.2.

You can re-create this experiment as follows;

```python
# TODO
```