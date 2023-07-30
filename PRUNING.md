# Model Pruning
To use purning techniques, you have to install package:
```
pip install torch-pruning
```
Prune model with the following command:
```
python prune_model.py -net resnet50 -weight checkpoint/resnet50/best-model.pth -speedup 2
```
The pruned model will be stored at **model.path**. To retrain pruned modle, you have to run the following command:
```
python train.py -net pruned -pruned_model model.pth -gpu
```
After retraining models, you can evaluate the pruned model with:
```
python test.py -net resnet50 -weight checkpoint/resnet50/pass.pth -pruned_path checkpoints/pruned-best.pth -gpu
```

