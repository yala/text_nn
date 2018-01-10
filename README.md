# text_nn

Text Classification models. Used as a submodule for other projects.
Supports extractive rationale extraction like in Tao Lei's Rationalizing neural prediction. These departs from Tao's
original frame work in the following way:

- I implement Generator training using the Gumbel Softmax instead of using REINFORCE
- I only implement the indepdent selector.


## Usage:
Example run:
```
CUDA_VISIBLE_DEVICES=2 python -u scripts/main.py  --batch_size 64 --cuda --dataset full_beer --embedding 
glove --dropout 0.05 --weight_decay 5e-06 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.0001 --num_workers
 0 --objective cross_entropy --patience 5 --save_dir snapshot --train --test --results_path logs/adhoc_0.results  --gumbel_decay 1e-5 --get_rationales
 --aspect aroma --selection_lambda .005 --continuity_lambda .01
```
Use `--get_rationales` to enable extractive rationales.

To run grid search, see `docs/dispatcher`

## Base Models Supported:
- TextCNN (Yoon 2014)

## Extending:
### How to add a new dataset:
- Add a pytorch Dataset object to `rationale_net/datasets` and register it to the dataset factory
### How to add a new model base?
- Supported in research version of this repo, but it's involved. If there is interest, please contact me.






