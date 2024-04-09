# Self-Attention GAN
**[Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena, "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)](https://arxiv.org/abs/1805.08318).**

# Improvement
We add the feature to use sparse attention instead of full attention algorithm. Specifically, we set the attention after softmax in a local of span x span square to be 0, which means don't allow a pixel to attend to its neighbors, because we believe convolution is able to extract local features efficiently.

## Results
### FID result on CelebA using span=0, which is origin work
<p align="center"><img width="80%" src="image/denormalized_raw.png" /></p>

### FID result on CelebA using span=1
<p align="center"><img width="80%" src="image/span=1.png" /></p>

### FID result on CelebA using span=2
<p align="center"><img width="80%" src="image/span=2.png" /></p>

### FID result on CelebA using span=3
<p align="center"><img width="80%" src="image/span=3.png" /></p>

### FID result on CelebA using span=4
<p align="center"><img width="80%" src="image/span=4.png" /></p>

It can be seen that our work converges faster than origin work, it takes more than 80,000 epochs for the origin work to reach FID lower than 30, but 40,000 epochs for our work to reach the same level



## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/zhengs81/SAGAN_PyTorch.git
$ cd SAGAN_PyTorch
```

#### 2. Install datasets (CelebA or LSUN)
```
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
or
https://www.kaggle.com/datasets/ajaykgp12/lsunchurch
```
Then move it under data/CelebA

#### 3. Train 
##### (i) Train without sparse attention，same result as origin work
```bash
$ python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb --total_step 100000 --span 0
```
##### (i) Train with sparse attention，our work
```bash
$ python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb --total_step 100000 --span 3
```
#### 4. Enjoy the results
```bash
$ cd sparse_samples_3
```
