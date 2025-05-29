# Distributed-Q-Algo
Distributed quantum algorithms.

## Install Locally (recommended)

1. Create your own conda environment with Python version == 3.12
```bash
conda create -n your_env_name python=3.12
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

## How to run tests

For one test:
```bash
pytest tests/test_nisq/test_fanout_by_ghz.py::test_truth_table_tomography -s
```

`-s` is to display `print` message in the test.

## How to run CSWAP circuit
To run the teledata scheme of the CSWAP run:
```
python ./scripts/eval_nisq_cswap.py --n_trgts 3 --p2 0.001

```

## Generate error distribution of Fanout

1. Generation
```bash
python scripts/eval_nisq_fanout.py -t 6 --p2 0.001 -s 1024
```
Read the script to see how to customize number of targets, error rates and shots.
The data will be saved at `data/nisq/fanout/` by `NISQFanoutDataMgr`.

2. Read

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# eval Baumer Fanout using stim's
Tableau` section.
```python
n_trgts_lst = [4, 6, 8]
p2s = [0.001, 0.003, 0.005]
n_shots = 100000

dmgr = NISQFanoutDataMgr()
data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
error_counts_lst, _, _ = data
print(error_counts_lst)
```

## Generate error distribution of Quantum Teleportation

1. Generation
```bash
python scripts/eval_nisq_teleport.py -t 6 --p2 0.001 -s 1024
```
Read the script to see how to customize number of targets, error rates and shots.
The data will be saved at `data/nisq/teleport/` by `NISQTeleportDataMgr`.

2. Read

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval Teleportation circuit using Stim.TableauSimulator` section.
```python
n_trgts_lst = [4, 6, 8]
p2s = [0.001, 0.003, 0.005]
n_shots = 100000

dmgr = NISQTeleportDataMgr()
data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
error_counts_lst, _, _ = data
print(error_counts_lst)
```

## Generate error distribution of telegate

1. Generation
```bash
python scripts/eval_nisq_telegate.py -t 6 --p2 0.001 -s 1024
```
Read the script to see how to customize number of targets, error rates and shots.
The data will be saved at `data/nisq/telegate/` by `NISQTelegateDataMgr`.

2. Read

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval telegate` section.
```python
n_trgts_lst = [4, 6, 8]
p2s = [0.001, 0.003, 0.005]
n_shots = 100000

dmgr = NISQTelegateDataMgr()
data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
error_counts_lst, _, _ = data
print(error_counts_lst)
```