## Basic array job example

This is a basic hyperparameter search example, intended for execution on a
GPU-endowed cluster like Graham or Cedar. It uses SLURM array jobs.

Before running, set up a basic virtualenv as follows:

```
$ module load python/3.6
$ mkdir venv
$ virtualenv vnev
$ source venv/bin/activate
(venv) $ pip install tensorflow-gpu keras --no-index
```

Then submit the training script as follows:

```
$ sbatch submit.sh
```


## Incremental architecture search

This example uses MPI to spawn groups of parallel training jobs, selecting
the best parameters from each group to determine the parameters for the next.

To run this example, set up the virtualenv as in the basic example. In
addition, install mpi4py:

```
(venv) $ pip install mpi4py --no-index
```

Then submit the job as follows:

```
$ sbatch ias_submit.sh
```
