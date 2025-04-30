# Distributed-Q-Algo
Distributed quantum algorithms.

## Install Locally (recommended)

1. Create your own conda environment with Python version == 3.12
```bash
conda create -n your_env_name python=3.12
```

2. Install dependencies:
```bash
pip install -r .
```

3. Git clone this repository, and `cd` into the folder, and install this package locally:
```bash
pip install -e .
```

4. Whenever activate this conda environment, you can use it as a Python package anywhere:
```bash
from dqalgo.nisq.fanouts import BaumerFanoutBuilder
```

5. Use in .ipynb and modify the code
If you modify the code in `/src` and want to test it in .ipynb, you need to 
add these two commands at the top cell of your .ipynb:
```bash
%load_ext autoreload
%autoreload 2
```

This will autoreload your modification.
