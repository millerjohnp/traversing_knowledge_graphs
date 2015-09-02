The repository contains the code, data, and parameters used in the following paper.

Kelvin Guu, John Miller, Percy Liang.
[Traversing Knowledge Graphs in Vector Space](http://arxiv.org/pdf/1506.01094.pdf)
Empirical Methods in Natural Language Processing (EMNLP), 2015.

If you use the dataset/code in your research, please cite the above paper.

	@inproceedings{gu2015traversing,
  		author = {K. Guu and J. Miller and P. Liang},
  		booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  		title = {Traversing Knowledge Graphs in Vector Space},
  		year = {2015},
	}
 
Reproducibility: A [Codalab](http://codalab.org) worksheet containing all of our experiments and several executable examples is available [here](https://www.codalab.org/worksheets/0xfcace41fdeec45f3bc6ddf31107b829f).

# Data
To automatically download the datasets used in our experiments, call the script ```get_datasets.sh.```

data/
 -   freebase/
      - train
	  - dev
	  - test
 -  wordnet/
	- train
	- dev
	- test

data format:
- Each line represents one _(source, relation, target)_ triple. Elements of the
triple are separated by tabs.
- In addition to test, we also include **test_induction** and **test_deduction**. These
correspond to the splits of the same name described in the paper.

To automatically download our parameters, call the script ```get_parameters.sh.```

params/
 -   freebase/
      - transE
	  - bilinear_diag
	  - bilinear
 -  wordnet/
	- train
	- dev
	- test
 
params format:
- Each file contains a pickled SparseVector object (pickled with cPickle).
- COMP files contain parameters that have been trained on the compositional
  dataset. SINGLE files contain parameters that have only been trained on 
  single edge queries.

# Code

To run an experiment using our code, call

    python demo.py CONFIG DATAPATH
 
from the code directory.
```CONFIG``` details the hyperparameters for the model and is defined in
```config.py.```  ```DATAPATH``` specifies a path to one of the datasets in
```data``` or your own dataset.
