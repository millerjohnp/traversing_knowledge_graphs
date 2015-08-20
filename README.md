Traversing Knowledge Graphs in Vector Space
Kelvin Gu, John Miller, Percy Liang
Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)

If you use the dataset/code in your research, please cite the above paper.

@article{gu2015traversing,
  title={Traversing Knowledge Graphs in Vector Space},
  author={Gu, Kelvin and Miller, John and Liang, Percy},
  journal={arXiv preprint arXiv:1506.01094},
  year={2015}
}

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
 
params/
    freebase/
        transE
        bilinear_diag
        bilinear
    wordnet/
        transE
        bilinear_diag
        bilinear

params format:
- Each file contains a pickled SparseVector object (pickled with cPickle).
- COMP files contain parameters that have been trained on the compositional
  dataset. SINGLE files contain parameters that have only been trained on 
  single edge queries.

# Code

To run an experiment using the code, call

    python demo.py CONFIG DATAPATH

```CONFIG``` details the hyperparameters for the model and is defined in
```config.py.```  ```DATAPATH``` specifies a path to one of the datasets in
```data``` or your own dataset.
