# PyTorch-finetune-transformer-lm

This is a PyTorch implementation of the [TensorFlow code](https://github.com/openai/finetune-transformer-lm) for OpenAI's paper "Improving Language Understanding by Generative Pre-Training" by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.

This implementation comprises **a script to load the weights pre-trained by the authors**.

The model classes and loading scripts are in the file [model_py.py](model_py.py). The name of the class instance follow the name used in the original implementation. The re-implementation also comprises an implementation of the modifier Adam optimization algorithm with fixed weights from OpenAI's paper.

## Requirements
For the model it-self in [model_py.py](model_py.py):
- pyTorch version >=0.4

Additional requirements to run the classifier training script in [train.py](train.py):
- tqdm
- sklearn
- spacy
- ftfy
- pandas

## Use the pre-trained model as a Transformer Language Model
The model can be used independently with the pre-trained weights by the following code:
```python
from model_py import Model, load_openai_pretrained_model, DEFAULT_CONFIG

args = DEFAULT_CONFIG,
vocab = 40000 # Size of your vocabulary
model = Model(vocab, args)
load_openai_pretrained_model(model)
```

You should encode your dataset using the `encode_dataset()` function of [utils.py](utils.py). Please refer to the beginning of the `__main__` function in [train.py](train.py) to see how to properly define your vocabulary and encode your dataset

## Fine-tune the pre-trained model on a classification task
The model can also be integrated in a classifier, for example to use it on the ROCStories Cloze Test task. Such an implementatino is detailed in the training code [train.py](train.py)

As the [TensorFlow code](https://github.com/openai/finetune-transformer-lm), this code implements the ROCStories Cloze Test result reported in the paper.

The ROCStories dataset can be downloaded from the associated [website](http://cs.rochester.edu/nlp/rocstories/).

This results can be reproduced by running:
`python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir [path to data here]`

## Note from OpenAI authors
The code is currently non-deterministic due to various GPU ops. The median accuracy of 10 runs with this codebase (using default hyperparameters) is 85.8% - slightly lower than the reported single run of 86.5% from the paper. 

### TO-DO list
- [ ] Add Multi-GPU training logic
