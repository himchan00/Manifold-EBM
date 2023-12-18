## Running
### 1. Train
```js
python train.py --config configs/mnist_eae_z2.yml --run mnist_ho_1 --device 0
```

- The result will be saved in './results' directory.  

### 2. Tensorboard 
```js
tensorboard --logdir results/
```
