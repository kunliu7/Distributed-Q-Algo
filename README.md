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
python ./scripts/eval_nisq_cswap.py --n_trgts 3 --p2 0.001 --scheme teledata

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


## Generate error distribution of parallel CNOTs

1. Generation
```bash
python scripts/eval_nisq_parallel_cnots.py -t 6 --p2 0.001 -s 1024
```
Read the script to see how to customize number of targets, error rates and shots.
The data will be saved at `data/nisq/parallel_cnots/` by `NISQParallelCNOTsDataMgr`.

2. Read

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval parallel CNOTs` section.
```python
n_trgts_lst = [4, 6, 8]
p2s = [0.001, 0.003, 0.005]
n_shots = 100000

dmgr = NISQParallelCNOTsDataMgr()
data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
error_counts_lst, _, _ = data
print(error_counts_lst)

```

## Running Simulations via SLURM

For large numbers of qubits (in my experience `n_targets > 4`), it may not be feasable to run `eval_nisq_cswap.py` on a local machine. Instead, it may be useful to run `eval_nisq_cswap_parallel.py` on a cluster via SLURM. An example SLURM script, `slurm_script_EXAMPLE.sh` is provided.

1. Edit the SLURM job configuration

Make a copy of `slurm_script_EXAMPLE.sh` in the root directory of the repository. At the top of the file there should be the configuration info:

```
#SBATCH --job-name=eval_nisq_cswap
#SBATCH --output=logs/eval_nisq_cswap_%j.out
#SBATCH --error=logs/eval_nisq_cswap_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-149
#SBATCH --mem=4G
```

Ensure that a the output/error folders exist or create one. You may also wish to modify the number of cpus (`--cpus-per-task`) and memory used (`--mem`) for larger jobs.The number of jobs and how they're indexed is controlled by the `--array` parameter. In our experiments, we used 150 samples so the default indexing is 0-149 but in the event that jobs fail during execution, you may need to adjust this.

2. Set the environment and path

Ensure that the line ```conda activate my_env``` (likely line 18) has the name corresponding to your environment.

Ensure that the line ```cd /your/project/directory``` (likely line 20) has the path to the root directory of the repository.

3. Set the simulation parameters

Next, find the line that runs the python script (likely line 23)

```python ./scripts/eval_nisq_cswap_parallel.py --n_trgts 4 --p2 0.001 --method telegate --output_file_prefix '../data/nisq/cswap/example_outputs_folder' --slurm_index $SLURM_ARRAY_TASK_ID```

and modify the arguments as desired.

3. Submit the job

Run the following line from the root directory of the respository:

```sbatch ./your_slurm_scrupt```

This will create 150 jobs, each sampling a single random input bitstring. Each job will create an output file in the specified output directory. On occasion, some jobs may fail due to memory constraints. In that case, you can run more jobs (be sure that the filenames don't overlap) until there are at least 150.

4. Compiling the data

We also provide a script which computes and prints the mean and standard deviation of the results files, which can be run via the following command:

```python ./scripts/compile_results.py --output_folder ./your/output/folder```




