# VIT-Lightning
An implementation of the Vision Transformer architecture from the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" using Pytorch Lightning

To run this repo, first install the dependencies using the ***requirements.txt*** (highly recommend to use environment manager like conda)
```
pip install -r requirements.txt
```

To start the training process, run
```
bash run.sh
```
In the ***run.sh*** script file, there are several settings can be added to the training process.
- --root_dir : directory to the training dataset, required.
- --test_dir : directory to the testing dataset, not required.
- --run : name of that training run, use for checkpointing location, required.
- --size : size of the image, default 224.
- --in_chans : number of image color channels, default 3 (RGB).
- --batch_size : size of each data batch, default 32.
- --lr : learning rate, default 1e-4.
- --epochs : number of training epochs, default 100.
- --patience : number of patience for lr scheduler and EarlyStopping callback, default 20.
- --arch : the size of the model backbone, including 'vit_tiny', 'vit_small' and 'vit_base', default 'vit_base'.
- --gpu_ids : the ids of the GPU used for training, default (0,) (GPU 0)
- --checkpoint_path : directory to the model checkpoints, default 'checkpoints'.
- --log_path : directory to the training logs, default '.' (current directory)