import torch
import torch_pruning as tp
from conf import settings
from utils import get_network, get_test_dataloader
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
parser.add_argument('-speedup', type=float, help='speed up model')
args = parser.parse_args()

model = get_network(args)
model.load_state_dict(torch.load(args.weights))
print(model)


# Importance criteria
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.MagnitudeImportance(p=1)

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 100:
        ignored_layers.append(m) # DO NOT prune the final classifier!

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
current_speed_up = 1
# print(model(example_inputs).shape)
while True:
    pruner.step()

    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    
    speed_up_time = base_macs / macs
    # print(model)
    print(model(example_inputs).shape)
    print(
        "  Speedup %.2f, Params: %.2f M => %.2f M"
        % (speed_up_time, base_nparams, nparams)
    )
    print(
        "  Speedup %.2f, MACs: %.2f G => %.2f G"
        % (speed_up_time, base_macs / 1e9, macs / 1e9)
    )
    print("=" * 16)

    if speed_up_time > args.speedup:
        break

model.zero_grad()  # We don't want to store gradient information
torch.save(model, 'model.pth')