## NMT decoding with recombination

Hi, this is an implementation of the paper "Exploring Recombination for Efficient Decoding of Neural Machine Translation".

## Usage

### Train:
Fisrt, we need a NMT model, use `train.py` to get it. Please refer to `examples/train.sh` as an example for the training of En-De and Zh-En models.

### Test:
Next, use the trained models for NMT decoding with `test.py`, see `examples/test.sh` as the reference.

### Options:
Please refer to `examples/options.md` for descriptions about cmd options.

### Requirements:
This code is implemented in python3 with [the DyNet toolkit](https://github.com/clab/dynet/). Please use a relatively newer version of DyNet, we tested it with the version of [this commit](https://github.com/clab/dynet/commit/e3bce8e54044b64a9990f506eba8652b11222afa).
