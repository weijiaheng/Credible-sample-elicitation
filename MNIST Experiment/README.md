# Credible sample elicitation

This code is the official Pytorch implementation of our paper "[Credible Sample Elicitation](https://arxiv.org/abs/1910.03155)" accepted by AISTATS2021.


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.



## Utilities

📋 (1) `dataset.py`: Prepare dataloader for MNIST dataset (load data from image folder);

📋 (2) `Inception_model.py`, `fid.py`, `fid_peer.py`: used for FID calculation, mainly adopted from `https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/inception.py`;

📋 (3) `f_div.py`: main part, including generating images, f-div score calculation (when there is ground-truth for verification).

📋 (4) `f_div_peer.py`: main part, including generating images, f-div score calculation (when there is no ground-truth for verification).



## To reproduce
> 📋Step 1:
Parameter/Argument setting:

epsilon | noise | divergence 
--- | --- | --- 
the misreport parameter | noise model type | f-divergence functions

Suggested setting:
epsilon: input a number between [0, 100] (will divided by 100);
noise: `PGDA` represents L_inf PGDAttack; `Gaussian` means Gaussian noise; `Speckle` means Speckle noise;
f-divergence functions: `Total-Variation` is the most stable choice. 
Provided divergences:   
Total-Variation | Jenson-Shannon | Pearson | KL | Jeffrey | Squared-Hellinger | Reverse-KL

> 📋Step 2: With ground-truth verification:
### To generate datasets and evaluate noise reports, run:
```
python f_div.py --gpu gpu_index --s seed --epsilon epsilon_parameter --noise 'noise_name' --divergence 'div_name'
```

### To calculate FID scores (assume `Total-Variation` ), run:
```
python fid.py --gpu gpu_index --batch-size batch_size --epsilon epsilon_parameter --noise 'noise_name' 
```

> 📋Step 3: Without ground-truth verification:
### To generate datasets and evaluate noise reports, run:
```
python f_div_peer.py --gpu gpu_index --s seed --epsilon epsilon_parameter --noise 'noise_name' --divergence 'div_name'
```

### To calculate FID scores (assume `Total-Variation` ), run:
```
python fid_peer.py --gpu gpu_index --batch-size batch_size --epsilon epsilon_parameter --noise 'noise_name' 
```

> 📋More details and hyperparameter settings can be seen in the supplementary materials and the corresponding runners.


