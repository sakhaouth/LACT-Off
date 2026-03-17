import torch
import argparse
import pandas as pd
from accelerate import Accelerator, DeepSpeedPlugin
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from torch import nn, optim
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser(description='Time-series-prediction')

parser.add_argument('--llm_model', type=str, default='GPT2')
parser.add_argument('--task_name', type=str, default="long_term_forecast")
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--llm_dim', type=int, default=768)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)

args = parser.parse_args()
accelerator = Accelerator()
for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    Time_LLM_model = TimeLLM.Model(args).float()
    args.content = load_content(args)
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    trained_parameters = []
    for p in Time_LLM_model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    time_llm_model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

configs = parser.parse_args()

model = TimeLLM(configs).float()

checkpoint = torch.load("model.pth", map_location=torch.device("cpu"))
model.projection.load_state_dict(checkpoint["projection"])
model.output_layer.load_state_dict(checkpoint["output_layer"])

# Set to evaluation mode if needed
model.eval()
df = pd.DataFrame()
tensor = torch.tensor(df.values, dtype=torch.float32)
tensor = tensor.unsqueeze(0)

