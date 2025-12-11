# COMPAS: A Distributed Multi-Party SWAP Test for Parallel Quantum Algorithms (ASPLOS 2026)

### [Brayden Goldstein-Gelb](http://brayden-gg.github.io), [Kun Liu](https://www.linkedin.com/in/kun-liu-0276141a4), [John M. Martyn](https://jmmartyn.github.io), [Hengyun Zhou](https://scholar.google.com/citations?user=XLHpQy8AAAAJ&hl=en), [Yongshan Ding](https://www.yongshanding.com), [Yuan Liu](https://ece.ncsu.edu/people/yliu335/)

We provide code to simulate the circuit constructions developed in the paper.
All the code, data and visualizations are in this repository.

**High-level overview:**
As mentioned in the paper, to simulate the circuit efficiently, we separate the overall circuit into multiple sub-circuits, and simulate each sub-circuit independently and derive their error distributions.
Then we can combine the error distributions of the sub-circuits to get the fidelity of the overall circuit.

**Hardware requirements:**
All the data used in the paper can be generated on a MacBook Pro with M1 chip and 16GB RAM.

## Installation instructions

### Install Locally (recommended)

1. Create your own conda environment with Python version 3.12
```bash
conda create -n your_env_name python=3.12
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Clone this repository, `cd` into the folder, and install this package locally:
```bash
pip install -e .
```

4. After activating this conda environment, you can use it as a Python package anywhere:
```bash
from dqalgo.nisq.fanouts import BaumerFanoutBuilder
```

5. Note about Jupyter notebooks

If you modify the code in the `/src` folder and want to test it in a jupyter notebook, you need to
add these two commands at the top cell of your notebook:
```bash
%load_ext autoreload
%autoreload 2
```

This will autoreload your modification.

### Run tests

To run all tests (this may take a while), run

```bash
pytest tests
```

For a single test function in a specific test file:
```bash
pytest tests/test_nisq/test_fanout_by_ghz.py::test_truth_table_tomography -s
```

`-s` is to display `print` message in the test.

## How to run the CSWAP circuits (top-level)

This is the top-level script to simulate the CSWAP circuits.

To run the teledata scheme of the CSWAP operation, run:
```
python ./scripts/eval_nisq_cswap.py --n_trgts 3 --p2 0.001 --method teledata --parallel
```

You can change the number of targets via the `--n_trgts` (`-t`) argument and set the method to either `teledata` or `telegate`.

Its components are simulated below.

### Generate the error distribution of a Fanout circuit

1. Generation
```bash
python ./scripts/eval_nisq_fanout.py -t 6 --p2 0.001 -s 1024
```
Here, `-t` is short for `--n_trgts` and `-s` or `--shots` can be used to set the number of shots. `--p2` represents the error probability of two-qubit gates (the 1-qubit error rate is set to `p2/10`)

2. View results

You can view the results in the included Jupyter notebook.
See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# eval Baumer Fanout using stim's
Tableau` section.

### Generate error distribution of Quantum Teleportation

1. Generation
```bash
python scripts/eval_nisq_teleport.py -t 6 --p2 0.001 -s 1024
```
As above,`-t` is short for `--n_trgts`, `-s`/`--shots` can be used to set the number of shots, and`--p2` represents the error probability of two-qubit gates.

2. View results

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval Teleportation circuit using Stim.TableauSimulator` section.


### Generate error distribution of a telegate operation

1. Generation
```bash
python scripts/eval_nisq_telegate.py -t 6 --p2 0.001 -s 1024
```

2. View results

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval telegate` section.


### Generate error distribution of parallel CNOTs

1. Generation
```bash
python scripts/eval_nisq_teleported_cnots.py -t 6 --p2 0.001 -s 1024
```

2. View results

See [notebooks/vis/eval_nisq_fanout.ipynb](notebooks/vis/eval_nisq_fanout.ipynb)'s `# Eval parallel CNOTs` section.

## Visualization

Figs. 9 and 10 used in the paper were generated using the following commands, which will take in the CSV files from the previous section (paths can also be provided as command line arguments) and save the resulting figures at the provided `--save_path` directory.

### Fig. 9(a)

```bash
python ./scripts/generate_ghz_graph.py --max_parties 12
```

### Fig. 9(b)

```bash
python ./scripts/generate_cswap_graph.py
```

### Fig. 9(c)

```bash
python ./scripts/generate_overall_error_graphs.py
```

### Fig. 10

```bash
python ./scripts/asymptotics_graphs.py
```

## Running Simulations via SLURM (optional)

If the number of targets is large, it may not be feasable to run `eval_nisq_cswap.py` on a local machine. Instead, it may be useful to run `eval_nisq_cswap_parallel.py` on a cluster via SLURM. An example SLURM script, `slurm_script_EXAMPLE.sh` is provided.

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

4. Submit the job

Run the following line from the root directory of the respository:

```sbatch ./your_slurm_script```

This will create 150 jobs, each sampling a single random input bitstring. Each job will create an output file in the specified output directory. On occasion, some jobs may fail due to memory constraints. In that case, you can run more jobs (be sure that the filenames don't overlap) until there are at least 150.

5. Compiling and plotting the data

We also provide a script which compiles the results and replaces it with a csv file:

```python ./scripts/compile_results.py --data_dir /path/to/data```