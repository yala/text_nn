# text_nn

Text Classification models. Used as a submodule for other projects.
Supports extractive rationale extraction like in Tao Lei's Rationalizing neural prediction. These departs from Tao's
original framework in the following way:

- I implement Generator training using the Gumbel Softmax instead of using REINFORCE
- I only implement the indepdent selector.

## Requirments
This repository assumes glove embeddings.
Download Glove embeddings at:  https://nlp.stanford.edu/projects/glove/
And place `glove.6B/glove.6B.300d.txt` in `data/embeddings/glove.6B/glove.6B.300d.txt`.

This code supports the the NewsGroup dataset and the BeerReview dataset. The for access to the BeerReview and the corresponding embeddings, please contact Tao Lei. I've included the NewsGroup dataset, conveiently provided by SKLearn so you can run code out of the box.


## Usage:
Example run:
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/main.py  --batch_size 64 --cuda --dataset news_group --embedding
glove --dropout 0.05 --weight_decay 5e-06 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.0001 --num_workers
 0 --objective cross_entropy --patience 5 --save_dir snapshot --train --test --results_path logs/demo_run.results  --gumbel_decay 1e-5 --get_rationales
 --selection_lambda .001 --continuity_lambda 0
```
Use `--get_rationales` to enable extractive rationales.

The results and extracted rationales will be saved in `results_path`
And be accessed as

```
results = pickle.load(open(results_path,'rb'))
rationales = results['test_stats']['rationales']
```

To run grid search, see `docs/dispatcher`.

Note, the rationale model is very sensitive to hyperparameters and the example run has not been tuned.


## Base Models Supported:
- TextCNN (Yoon 2014)

## Extending:
### How to add a new dataset:
- Fork the repo 
- Add a pytorch Dataset object to `rationale_net/datasets` and register it to the dataset factory. See the news_group and beer_review datasets for an example.
- Add the corresponding import to `rationale_net/datasets/__init__.py`
### How to add a new model base?
- Supported in research version of this repo, but it's involved. If there is interest, please contact me.






