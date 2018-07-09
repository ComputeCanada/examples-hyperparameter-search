This is a basic hyperparameter search example, intended for execution on a
GPU-endowed cluster like Graham or Cedar.

Before running, set up a basic virtualenv as follows:

$ module load python/3.6
$ mkdir venv
$ virtualenv vnev
$ source venv/bin/activate
(venv) $ pip install tensorflow-gpu keras --no-index

Then submit the training script as follows:

$ sbatch submit.sh
