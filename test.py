
from New_Time_LLM import Model
import torch
import argparse
import pandas as pd
from accelerate import Accelerator, DeepSpeedPlugin
from data_provider.data_factory import data_provider
from torch import nn, optim
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import os
from LSTM import MultiValueLSTM
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
if __name__ == "__main__":
    print("Starting predictor training")

    parser = argparse.ArgumentParser(description='Time-series-prediction')

    # basic config
    parser.add_argument('--llm', type=int, required=True, default=1, help='llm should train')
    parser.add_argument('--lstm', type=int, required=True, default=1, help='lstm should train')
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
    def vali(test_loader, model, criterion, mae_metric, device):
        total_loss = []
        total_mae_loss = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x)
                pred = outputs.detach()
                true = batch_y
                loss = criterion(pred, true)
                mae_loss = mae_metric(pred, true)
                total_loss.append(loss.item())
                total_mae_loss.append(mae_loss.item())
            
        total_loss = np.average(total_loss)
        total_mae_loss = np.average(total_mae_loss)
        model.train()
        return total_loss, total_mae_loss
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, seq_len=16, pred_len = 1):
            self.data = torch.tensor(data.values, dtype=torch.float32)
            self.seq_len = seq_len
            self.pred_len = pred_len
            
        def __len__(self):
            # total possible sequences
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            # print(f'idx is {idx}')
            # x: previous 5 rows
            x = self.data[idx: idx + self.seq_len]

            # y: next row
            y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

            # y = y.unsqueeze(1)

            return x, y

    df = pd.read_csv("ETTh1.csv")
    df = df.drop(columns=["date"])
    split_value = 1000
    train_data = df[:split_value]
    test_data = df[split_value:]
    scaler = StandardScaler()

    train_st = pd.DataFrame(
        scaler.fit(train_data),   # scale
        columns=train_data.columns,
        index=train_data.index

    )
    train_data = pd.DataFrame(
        scaler.transform(train_data),
        columns=train_data.columns,
        index=train_data.index
    )
    train_data = train_data[1:]
    test_st = pd.DataFrame(
        scaler.fit(test_data),   # scale
        columns=test_data.columns,
        index=test_data.index

    )
    test_data = pd.DataFrame(
        scaler.transform(test_data),
        columns=test_data.columns,
        index= test_data.index
    )

    # print(train_data)
    dataset = TimeSeriesDataset(train_data, seq_len=16)
    test_dataset = TimeSeriesDataset(test_data, seq_len=16)
    batch_size = 16
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    num_feature = train_data.shape[1]
    time_llm_model = Model(args).float()
    trained_parameters = []
    for p in time_llm_model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    llm_optimizer = optim.Adam(trained_parameters, lr=args.llm_learning_rate)
    model = MultiValueLSTM(input_size= num_feature , output_size= num_feature)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 100
    df = pd.DataFrame(columns=["lstm_train_loss", "lstm_test_loss", "lstm_test_mae", "llm_train_loss", "llm_test_loss", "llm_test_mae"])
    train_steps = len(train_loader)
    # lstm_scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                # steps_per_epoch=train_steps,
                                                # pct_start = 0.2,
                                                # epochs=epoch,

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    time_llm_model = time_llm_model.to(device)  
    # optimizer.to(device)
    # llm_optimizer.to(device)                                          # max_lr=0.001)
    for e in range(epoch):
        model.train()
        lstm_total_loss = 0
        
        llm_total_loss = 0
        for batch_x, batch_y in train_loader:
            # print("iteration")
            optimizer.zero_grad()
            llm_optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            # print("New data")
            # print(batch_x)  # [10, 5, features]
            # print(batch_y)  # [10, features]
            if args.lstm == 1:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                # lstm_scheduler.step()

                lstm_total_loss += loss.item()
            if args.llm == 1:
                outputs = time_llm_model(batch_x)
                llm_loss = criterion(outputs, batch_y)
                llm_loss.backward()
                llm_optimizer.step()
                llm_total_loss += llm_loss
        lstm_vali_loss =  lstm_vali_mae = llm_vali_loss = llm_vali_mae = None
        if args.lstm == 1:
            lstm_vali_loss, lstm_vali_mae = vali(test_loader, model, criterion, mae_metric,device)
        if args.llm == 1:
            llm_vali_loss, llm_vali_mae = vali(test_loader, time_llm_model, criterion, mae_metric, device)
        # row = [total_loss, valis_loss, vali_mae]
        row = [lstm_total_loss, lstm_vali_loss, lstm_vali_mae, llm_total_loss, llm_vali_loss, llm_vali_mae]
        df.loc[len(df)] = row
        print(f"Epoch {e+1}, Loss: {lstm_total_loss}, Vali = {lstm_vali_loss, lstm_vali_mae}")
        # print('Updating learning rate to {}'.format(lstm_scheduler.get_last_lr()[0]))

    df.to_csv("test_log.csv", index=True)

