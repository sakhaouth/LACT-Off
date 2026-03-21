import torch
import argparse
import pandas as pd
from accelerate import Accelerator, DeepSpeedPlugin
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from torch import nn, optim
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import os
from LSTM import MultiValueLSTM
import pandas as pd


print("Starting predictor training")

parser = argparse.ArgumentParser(description='Time-series-prediction')

# basic config

parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
#------------------------------------------------------------------------------------

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#------------------------------------------------------------------------------------
# forecasting task
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=16)
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--llm_learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lstm_learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
accelerator = Accelerator()
columns = ["Train Loss", "Vali Loss", "Test Loss", "MAE Loss"]
dataFrame = pd.DataFrame(columns= columns)
for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.llm_model,
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
    LSTM_model = MultiValueLSTM(pred_len=args.pred_len)
    args.content = load_content(args)
    train_steps = len(train_loader)
    time_llm_early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    lstm_early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    trained_parameters = []
    for p in Time_LLM_model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    lstm_model_optim = optim.Adam(LSTM_model.parameters(), lr=args.lstm_learning_rate)
    time_llm_model_optim = optim.Adam(trained_parameters, lr=args.llm_learning_rate)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(time_llm_model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=time_llm_model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.llm_learning_rate)
        lstm_scheduler = lr_scheduler.OneCycleLR(optimizer=lstm_model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.lstm_learning_rate)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    train_loader, vali_loader, test_loader, Time_LLM_model, time_llm_model_optim,LSTM_model,lstm_model_optim, scheduler, lstm_scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, Time_LLM_model, time_llm_model_optim,LSTM_model,lstm_model_optim, scheduler, lstm_scheduler)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        time_llm_train_loss = []
        lstm_train_loss = []
        LSTM_model.train()
        Time_LLM_model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            time_llm_model_optim.zero_grad()
            lstm_model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            # print(f'info it {i}, shape {batch_x.shape} {batch_y.shape}')
            # decoder input
            #----------------------------TIME-LLM--------------------------------
            # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
            #     accelerator.device)
            # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
            #     accelerator.device)
            # time_llm_outputs = Time_LLM_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # f_dim = -1 if args.features == 'MS' else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -args.pred_len:, f_dim:]
            # time_llm_loss = criterion(outputs, batch_y)
            # time_llm_train_loss.append(time_llm_loss.item())
            # accelerator.backward(time_llm_loss)
            # time_llm_model_optim.step()
            #----------------------------LSTM--------------------------------
            
            lstm_output = LSTM_model(batch_x)
            # print(f'shape are {lstm_output.shape}, {batch_y.shape}')
            lstm_loss = criterion(lstm_output, batch_y)
            lstm_train_loss.append(lstm_loss.item())
            accelerator.backward(lstm_loss)
            torch.nn.utils.clip_grad_norm_(LSTM_model.parameters(), 1.0)
            lstm_model_optim.step()
            lstm_scheduler.step()
        #-----------------------------------Time-LLM-------------------------------------
        # time_llm_train_loss_av = np.average(time_llm_train_loss)
        # time_llm_vali_loss, time_llm_vali_mae_loss = vali(args, accelerator, Time_LLM_model, vali_data, vali_loader, criterion, mae_metric)
        # time_llm_test_loss, time_llm_test_mae_loss = vali(args, accelerator, Time_LLM_model, test_data, test_loader, criterion, mae_metric)
        # path = os.path.join(args.checkpoints,
        #                 setting + '-' + args.model_comment)
        # time_llm_early_stopping(time_llm_vali_loss, Time_LLM_model, path)
        # if args.lradj != 'TST':
        #     if args.lradj == 'COS':
        #         scheduler.step()
        #         print("lr = {:.10f}".format(time_llm_model_optim.param_groups[0]['lr']))
        #     else:
        #         if epoch == 0:
        #             args.llm_learning_rate = time_llm_model_optim.param_groups[0]['lr']
        #             print("lr = {:.10f}".format(time_llm_model_optim.param_groups[0]['lr']))
        #         adjust_learning_rate(accelerator, time_llm_model_optim, scheduler, epoch + 1, args, printout=True)

        # else:
        #     print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        #--------------------------------------------LSTM--------------------
        lstm_train_loss_av = np.average(lstm_train_loss)
        print(f'Epoch loss is {lstm_train_loss_av}')
        lstm_vali_loss, lstm_vali_mae_loss = vali(args, accelerator, LSTM_model, vali_data, vali_loader, criterion, mae_metric,True)
        lstm_test_loss, lstm_test_mae_loss = vali(args, accelerator, LSTM_model, test_data, test_loader, criterion, mae_metric, True)
        row = [lstm_train_loss_av, lstm_vali_loss, lstm_test_loss, lstm_test_mae_loss]
        dataFrame.loc[len(dataFrame)] = row
        print(f'Statistics is: lstm_vali_loss, lstm_vali_mae_loss {lstm_vali_loss, lstm_vali_mae_loss} and lstm_test_loss, lstm_test_mae_loss are {lstm_test_loss, lstm_test_mae_loss}')
        path = os.path.join(args.checkpoints,
                        setting + '-lstm-' + args.model_comment)
        os.makedirs(path, exist_ok=True)
        lstm_early_stopping(lstm_vali_loss, LSTM_model, path)
        # if lstm_early_stopping.early_stop:
        #     print("Early stopping")
            
        #     break
        if args.lradj != 'TST':
            if args.lradj == 'COS':
                # lstm_scheduler.step()
                print("lr = {:.10f}".format(lstm_model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.lstm_learning_rate = lstm_model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(lstm_model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, lstm_model_optim, lstm_scheduler, epoch + 1, args, printout=True)

        else:
            print('Updating learning rate to {}'.format(lstm_scheduler.get_last_lr()[0]))
accelerator.wait_for_everyone()
dataFrame.to_csv("log.csv", index=True)
if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    print('success delete checkpoints')

    

# configs = parser.parse_args()

# model = TimeLLM(configs).float()

# checkpoint = torch.load("model.pth", map_location=torch.device("cpu"))
# model.projection.load_state_dict(checkpoint["projection"])
# model.output_layer.load_state_dict(checkpoint["output_layer"])

# # Set to evaluation mode if needed
# model.eval()
# df = pd.DataFrame()
# tensor = torch.tensor(df.values, dtype=torch.float32)
# tensor = tensor.unsqueeze(0)

