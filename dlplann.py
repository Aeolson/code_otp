import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import random
import copy
import time
from rich import print
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from thop import profile
import datci
from configs import *
from utils.simple_agent import Agent, AgentTrajectory, get_agent_from_daci

use_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hist_seqlen = 10 # 1.0s
pred_seqlen = 40 # 4.0s
batch_size = 64
# a_min, a_max = -2.0, 4.0
# d_min, d_max = np.tan(np.deg2rad(-25)), np.tan(np.deg2rad(25))

def get_state_by_control(s:np.ndarray, u:np.ndarray, Ts:float):
    out_state = np.zeros((len(u),4), float)
    for k in range(len(u)):
        if k == 0:
            x, y, h, v = s
        else:
            x, y, h, v = out_state[k-1]

        acc = u[k,0] * a_max if u[k,0] > 0 else u[k,0] * abs(a_min)
        dfw = u[k,1] * d_max if u[k,1] > 0 else u[k,1] * abs(d_min)
        # acc, dfw = u[k,:]
        
        # out_state[k] =[
        #     x + Ts * v * np.cos(h),
        #     y + Ts * v * np.sin(h),
        #     h + Ts * v / wheelbase * dfw,
        #     v + Ts * acc,
        # ]
        out_state[k] =[
            x + Ts * v * np.cos(h),
            y + Ts * v * np.sin(h),
            h + Ts * v / wheelbase * dfw,
            v + Ts * acc,
        ]
    return out_state

class Dataset:
    def __init__(self, use_ds, train_ratio:float=None) -> None:
        datci.createReplayDataset(use_ds, resampleRate=round(1.0/simu_pace))
        datci.setSimupace(simu_pace)

        self.lane_change_infos = self.load_lane_change_infos(use_ds)
        if train_ratio is not None:
            assert isinstance(train_ratio, float) and train_ratio > 0.0 and train_ratio < 1.0
            n_train = round(len(self.lane_change_infos) * train_ratio)
            random.shuffle(self.lane_change_infos)
            self.train_data = self.create_data(self.lane_change_infos[:n_train])
            self.evalu_data = self.create_data(self.lane_change_infos[n_train:])
        else:
            self.evalu_data = self.create_data(self.lane_change_infos)

    def load_lane_change_infos(self, use_ds):
        lane_change_infos: list[dict] = []
        LC_INFO_FILE = "./LC_INFOS/LC_INFO_%s.txt" % (use_ds['name'])
        with open(LC_INFO_FILE, 'r') as fr:
            ls = fr.readlines()
            for l in ls:
                l = l.replace('\n','')
                lc = {}
                for s in l.split(','):
                    k_, v_ = s.split(':')
                    if k_ in ['t_cross', 't_start', 't_end']:
                        v_ = int(v_)
                    if k_ in ['lane_from_centery', 'lane_from_width', 'lane_to_centery', 'lane_to_width']:
                        v_ = float(v_)
                    if v_ == 'None':
                        v_ = None
                    lc[k_] = v_
                lane_change_infos.append(lc)
        
        return lane_change_infos

    def create_data(self, infos:list[dict]) -> list[np.ndarray]:
        if len(infos) == 0:
            return []
        
        datas, masks = [], []
        for lc in tqdm(infos):
            ego_id = lc['ego']
            t_start, t_end = lc['t_start'], lc['t_end']
            lane_from, lane_from_centery, lane_from_width = lc['lane_from'], lc['lane_from_centery'], lc['lane_from_width']
            lane_to, lane_to_centery, lane_to_width = lc['lane_to'], lc['lane_to_centery'], lc['lane_to_width']
            n_frame = t_end - t_start + 1
            data = np.zeros((n_frame, 5, 4), float) # (n_frame, n_veh, n_feature)
            mask = np.zeros((n_frame, 5), bool)     # (n_frame, n_veh)
            vids = np.zeros((n_frame, 5), int)     
            for k in range(n_frame):
                datci.setTimestep(t_start + k)
                all_vehicles = datci.getFrameVehicleIds()
                
                veh_e: Agent = get_agent_from_daci(ego_id) # ego vehicle
                veh_p: Agent = None # preceding vehicle in the original lane
                veh_b: Agent = None # back vehicle in the original lane
                veh_f: Agent = None # front vehicle in the target lane
                veh_r: Agent = None # rear vehicle in the target lane

                for Vid in all_vehicles:
                    if Vid == veh_e.id:
                        continue
                    Vlane = datci.vehicle.getLaneID(Vid)
                    if Vlane == lane_from:
                        Vx, Vy = datci.vehicle.getPosition(Vid)
                        if Vx > veh_e.x:
                            if veh_p is None or veh_p.x > Vx:
                                veh_p = get_agent_from_daci(Vid)
                        else:
                            if veh_b is None or veh_b.x < Vx:
                                veh_b = get_agent_from_daci(Vid)
                    elif Vlane == lane_to:
                        Vx, Vy = datci.vehicle.getPosition(Vid)
                        if Vx > veh_e.x:
                            if veh_f is None or veh_f.x > Vx:
                                veh_f = get_agent_from_daci(Vid)
                        else:
                            if veh_r is None or veh_r.x < Vx:
                                veh_r = get_agent_from_daci(Vid)

                # if veh_r.id == '352':
                #     print("[yellow] k=%d, id=%s, x=%.1f, y=%.1f, v=%.1f, h=%.1f[/yellow]" % (
                #         t_start+k, veh_r.id, veh_r.x, veh_r.y, veh_r.v, veh_r.h
                #     ))
                
                data[k] = [
                    [V.x, V.y, V.h, V.v] if V is not None else [0,0,0,0] for V in [veh_e, veh_p, veh_b, veh_f, veh_r]
                ]
                mask[k] = [
                    True if V is not None else False for V in [veh_e, veh_p, veh_b, veh_f, veh_r]
                ]
                vids[k] = [
                    int(V.id) if V is not None else 0 for V in [veh_e, veh_p, veh_b, veh_f, veh_r]
                ]

            control_inputs = np.zeros((n_frame, 2), float)
            Ts = simu_pace
            sx, sy, sh, sv = data[0,0,:]
            for k in range(1, n_frame):
                tx, ty, th, tv = data[k,0,:]
                # th = np.arctan2(ty-sy, tx-sx)
                # tv = np.sqrt((ty-sy)**2 + (tx-sx)**2) / Ts

                ak = (tv - sv) / Ts
                dk1 = (ty - sy - Ts * sv * sh) * wheelbase / wheelbase_r / max(sv, 1.0) / Ts
                dk2 = (th - sh) * wheelbase / max(sv, 1.0) / Ts
                alpha = 0.2
                dk = alpha * dk1 + (1-alpha) * dk2
                # dk = dk2

                ak = max(a_min, min(a_max, ak))
                dk = max(d_min, min(d_max, dk))

                sx = sx + Ts * sv * np.cos(sh)
                sy = sy + Ts * sv * np.sin(sh)
                sh = sh + Ts * sv / wheelbase * dk
                sv = sv + Ts * ak

                ak = ak / a_max if ak > 0 else ak / abs(a_min)
                dk = dk / d_max if dk > 0 else dk / abs(d_min)
                control_inputs[k] = [ak, dk]
                data[k,0,:] = [sx, sy, sh, sv]
            
            show = False
            if show:
                print(control_inputs[:,0].round(2))
                print(control_inputs[:,1].round(2))
                print("-----------------------")
                plt.figure(1)
                plt.plot(data[:,0,0], data[:,0,1], c='r', ls='-', lw=2.0)
                out = get_state_by_control(data[0,0,:], control_inputs, simu_pace)
                plt.plot(out[:,0], out[:,1], c='g', ls=':', lw=2.0)
                plt.show()

            for k in range(hist_seqlen, n_frame-pred_seqlen+1):
                hist_states = copy.deepcopy(data[k-hist_seqlen:k, :, :])
                pred_xys = copy.deepcopy(data[k:k+pred_seqlen, 0, :2])

                # reset trajectory original
                x0 = hist_states[-1,0,0]
                y0 = lane_to_centery

                hist_states[:,:,0] -= x0
                hist_states[:,:,1] -= y0

                pred_xys[:,0] -= x0
                pred_xys[:,1] -= y0

                pred_ctrls = control_inputs[k:k+pred_seqlen,:]

                m = mask[k-hist_seqlen:k]
                hist_states[m==False] = np.zeros(hist_states.shape[-1])
                datas.append([hist_states, pred_xys, pred_ctrls])
        
        return datas

class EarlyStopping(object):
    
    def __init__(self, patience=5, verbose=False, delta=0):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True, 为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def check(self, epoch, val_loss):

        score = val_loss

        if self.best_epoch is None or score < self.best_score - self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.early_stop = False

        else:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        print("Checkpoint: epoch = %4d, loss = %.4f, best_epoch = %4d, best_loss = %.4f"%(epoch, score, self.best_epoch, self.best_score))

class attention_scaled_dot_product(nn.Module):
    def __init__(self, scale=1.0) -> None:
        super(attention_scaled_dot_product, self).__init__()
        self.scale = scale

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None):
        '''
        forward of scaled-dot-product in attention
        q.shape = (n_batch, n_q, dim)
        k.shape = (n_batch, n_k, dim)
        v.shape = (n_batch, n_v, dim)
        mask = None: no mask
        scale = 1.0: dot-product
        dropout = 0.0: no dropout
        return: (context, softmax output) 
        '''
        att_alpha = torch.matmul(q, k.permute(0, 2, 1)) / np.sqrt(self.scale)
        if mask is not None:
            att_alpha = att_alpha.masked_fill(mask, -1e10)
        att_alpha = torch.softmax(att_alpha, dim=-1)
        context = torch.matmul(att_alpha, v)

        return context, att_alpha

class multi_head_attention(nn.Module):
    def __init__(self, in_feature, att_feature, n_head) -> None:
        super(multi_head_attention, self).__init__()
        self.in_feature = in_feature
        self.att_feature = att_feature
        self.out_feature = n_head * att_feature
        self.n_head = n_head

        self.att_Q = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.att_K = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.att_V = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.dot_product = attention_scaled_dot_product(scale=att_feature)
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None):
        n_batch = q.shape[0]
        n_q = q.shape[1]
        n_k = k.shape[1]
        n_v = v.shape[1]
        
        vec_q = self.att_Q(q)   # n_batch x n_q x n_head*dim
        arr_K = self.att_K(k)   # n_batch x n_k x n_head*dim
        arr_V = self.att_V(v)   # n_batch x n_v x n_head*dim
        
        vec_q = vec_q.reshape(n_batch, n_q, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_q, self.att_feature)
        arr_K = arr_K.reshape(n_batch, n_k, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_k, self.att_feature)
        arr_V = arr_V.reshape(n_batch, n_v, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_v, self.att_feature)
        if mask is not None:
            arr_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).reshape(n_batch*self.n_head, n_q, n_k)
        else:
            arr_mask = None

        att_output, att_alpha = self.dot_product(vec_q, arr_K, arr_V, arr_mask)
        att_output = att_output.reshape(n_batch, self.n_head, n_q, self.att_feature).permute(0,2,1,3).reshape(n_batch, n_q, self.out_feature)
        att_alpha = att_alpha.reshape(n_batch, self.n_head, n_q, n_k).permute(0,2,1,3)

        # print(q.shape, vec_q.shape, arr_K.shape, att_output.shape)

        return att_output, att_alpha

class ASLSTM(nn.Module):
    def __init__(self) -> None:
        super(ASLSTM, self).__init__()

        self.in_features  = 4 # [x, y, h, v]
        self.out_features = 2 # [x, y]

        # input embeding
        self.inp_emb_dim = 64
        self.inp_emb = nn.Linear(in_features=self.in_features, out_features=self.inp_emb_dim)
        
        # encoder-attention
        self.enc_att_num = 4
        self.enc_att_dim = 32
        self.enc_att = multi_head_attention(in_feature=self.inp_emb_dim, att_feature=self.enc_att_dim, n_head=self.enc_att_num)
        self.enc_att_out = nn.Linear(in_features=self.enc_att_num*self.enc_att_dim, out_features=self.inp_emb_dim)
        # encoder-mlp
        self.enc_mlp_upf = nn.Linear(in_features=self.inp_emb_dim, out_features=4*self.inp_emb_dim)
        self.enc_mlp_dwf = nn.Linear(in_features=4*self.inp_emb_dim, out_features=self.inp_emb_dim)

        # social-attention
        self.soc_att_num = 4
        self.soc_att_dim = 32
        self.soc_att = multi_head_attention(in_feature=self.inp_emb_dim, att_feature=self.soc_att_dim, n_head=self.soc_att_num)
        self.soc_att_out = nn.Linear(in_features=self.soc_att_num*self.soc_att_dim, out_features=self.inp_emb_dim)
        # social-mlp
        self.soc_mlp_upf = nn.Linear(in_features=self.inp_emb_dim, out_features=4*self.inp_emb_dim)
        self.soc_mlp_dwf = nn.Linear(in_features=4*self.inp_emb_dim, out_features=self.inp_emb_dim)

        # LSTM-based decoder
        self.dec_lstm = nn.LSTM(input_size=self.inp_emb_dim, hidden_size=128, batch_first=True)
        self.dec_output = nn.Linear(in_features=self.dec_lstm.hidden_size, out_features=self.out_features)

        # Activations:
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, hist_data:Tensor) -> Tensor:
        hist_data = hist_data.transpose(1,2)
        n_batch, n_veh, n_seq, n_feature = hist_data.shape
        
        tens_inp_tag = hist_data[:,:1,:,:].clone()                                       # n_batch x 1 x n_seq x n_in_feature
        tens_inp_ngh = hist_data[:,1:,:,:].clone()                                       # n_batch x 4 x n_seq x n_in_feature
        tens_pos_tag = hist_data[:,:1,:,:2].clone()                                      # n_batch x 1 x n_seq x 2
        tens_pos_ngh = hist_data[:,1:,:,:2].clone()                                      # n_batch x 4 x n_seq x 2
        tens_inp_all = torch.cat((tens_inp_tag, tens_inp_ngh), dim=1)                    # n_batch x 5 x n_seq x n_in_feature

        # input embeding
        tens_inp_emb = self.leaky_relu(self.inp_emb(tens_inp_all))                       # n_batch x 5 x n_seq x n_inp_emb_dim

        # position encoding
        t_ = np.array([np.power(3.0, 2.0*(i_//2)/self.inp_emb_dim) for i_ in range(self.inp_emb_dim)])
        tens_pe = np.array([p_/t_ for p_ in range(n_seq)])
        tens_pe[:, 0::2] = np.sin(tens_pe[:, 0::2])
        tens_pe[:, 1::2] = np.cos(tens_pe[:, 1::2])
        tens_pe = torch.from_numpy(tens_pe.copy()).float().to(use_device)
        tens_pe = tens_pe.repeat(n_batch, n_veh, 1, 1)
        tens_inp_emb = tens_inp_emb + tens_pe                                           # n_batch x 5 x n_seq x n_inp_emb_dim

        # attention-based encoder
        tens_enc_q = tens_inp_emb[:,:,-1:,:].reshape(n_batch*n_veh, 1, self.inp_emb_dim)        # n_batch*5 x 1       x n_inp_emb_dim
        tens_enc_k = tens_inp_emb[:,:,:-1,:].reshape(n_batch*n_veh, n_seq-1, self.inp_emb_dim)  # n_batch*5 x n_seq-1 x n_inp_emb_dim
        tens_enc_v = tens_inp_emb[:,:,:-1,:].reshape(n_batch*n_veh, n_seq-1, self.inp_emb_dim)  # n_batch*5 x n_seq-1 x n_inp_emb_dim

        tens_enc_att, _ = self.enc_att(tens_enc_q, tens_enc_k, tens_enc_v)                  # n_batch*5 x 1 x n_enc_att_num*n_enc_att_dim
        tens_enc_out = self.enc_att_out(tens_enc_att).reshape(n_batch, n_veh, self.inp_emb_dim) # n_batch x 5 x n_inp_emb_dim
        tens_enc_out = self.leaky_relu(tens_enc_out) + tens_inp_emb[:,:,-1,:]

        tens_enc_mlp = self.enc_mlp_dwf(self.leaky_relu(self.enc_mlp_upf(tens_enc_out)))    # n_batch x 5 x n_inp_emb_dim
        tens_enc_mlp = self.leaky_relu(tens_enc_mlp) + tens_enc_out


        # attention-based social interaction
        tens_soc_q = tens_enc_mlp[:,:1,:]                                                   # n_batch x 1 x n_inp_emb_dim
        tens_soc_k = tens_enc_mlp[:,1:,:]                                                   # n_batch x 4 x n_inp_emb_dim
        tens_soc_v = tens_enc_mlp[:,1:,:]                                                   # n_batch x 4 x n_inp_emb_dim

        # attention mask
        tens_mask_ngh = torch.zeros((n_batch, n_veh-1)).int().to(use_device)
        mask_ = torch.sum(torch.abs(tens_pos_ngh[:,:,-1,:]), dim=-1) < 0.1
        tens_mask_ngh = tens_mask_ngh.masked_fill(mask_, 1)
        # mask_ = torch.abs(tens_pos_ngh[:,:,-1,0] - tens_pos_tag[:,0:1,-1,0]) > 50.0
        # tens_mask_ngh = tens_mask_ngh.masked_fill(mask_, 1)
        tens_mask_ngh = tens_mask_ngh.bool().unsqueeze(1)

        tens_soc_att, _ = self.soc_att(tens_soc_q, tens_soc_k, tens_soc_v, tens_mask_ngh)   # n_batch x 1 x n_soc_att_num*n_soc_att_dim
        tens_soc_out = self.soc_att_out(tens_soc_att).reshape(n_batch, self.inp_emb_dim)    # n_batch x n_inp_emb_dim
        tens_soc_out = self.leaky_relu(tens_soc_out) + tens_enc_mlp[:,0,:]
        
        tens_soc_mlp = self.soc_mlp_dwf(self.leaky_relu(self.soc_mlp_upf(tens_soc_out)))    # n_batch x n_inp_emb_dim
        tens_soc_mlp = self.leaky_relu(tens_soc_mlp) + tens_soc_out

        # social mask
        tens_mask_soc = torch.sum(tens_mask_ngh, dim=-1)
        tens_mask_soc = (tens_mask_soc == n_veh-1).repeat(1, self.inp_emb_dim).int().float()
        tens_soc_mlp = tens_soc_mlp * (1-tens_mask_soc)

        # LSTM-based decoder
        tens_dec_inp = tens_enc_mlp[:,0,:] + tens_soc_mlp                                       # n_batch x n_inp_emb_dim
        tens_dec_inp = tens_dec_inp.unsqueeze(-2).repeat(1, pred_seqlen, 1)             # n_batch x n_predlen x n_inp_emb_dim
        tens_dec_out, (_, _) = self.dec_lstm(tens_dec_inp)
        tens_dec_out = self.dec_output(self.leaky_relu(tens_dec_out))                           # n_batch x n_predlen x 2
                        
        return torch.tanh(tens_dec_out)

def lossfuc(pl:Tensor, gt:Tensor) -> Tensor:
    # err = (pl - gt) / 0.1
    # return torch.sum(torch.pow(err, 2)) / len(gt)
    err = pl - gt
    return torch.sum(torch.exp(torch.sum(torch.pow(err, 2), dim=-1))) / len(gt)

class LaneChangePlanner_DLP:
    def __init__(self, mdlfile:str = None) -> None:
        self.model = ASLSTM()
        if mdlfile is not None:
            try:
                self.model.load_state_dict(torch.load(mdlfile))
            except:
                print("[red]model is not loadded successfully !!![/red]")
        
        self.veh_qlist: list[tuple[Agent]] = []
    
    def train(self):
        model = self.model.to(use_device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        stopper = EarlyStopping(patience=10, delta=0.0)
        dataset = Dataset(use_ds=datci.NGSIM_I80_0400_0415, train_ratio=0.8)
        train_data, evalu_data = dataset.train_data, dataset.evalu_data
        n_batch_train = len(train_data) // batch_size if len(train_data) % batch_size == 0 else len(train_data) // batch_size + 1
        n_batch_evalu = len(evalu_data) // batch_size if len(evalu_data) % batch_size == 0 else len(evalu_data) // batch_size + 1

        for e in range(1000):
            random.shuffle(train_data)
            model.train()
            for b in range(n_batch_train):
                batch_data = train_data[b*batch_size:(b+1)*batch_size]
                hist_data = [d[0] for d in batch_data]
                futu_traj = [d[1] for d in batch_data]
                hist_data = torch.from_numpy(np.array(hist_data)).float().to(use_device)
                futu_traj = torch.from_numpy(np.array(futu_traj)).float().to(use_device)

                plan_traj = model.forward(hist_data)
                loss_ = lossfuc.forward(plan_traj, futu_traj)

                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
            
            model.eval()
            evalu_loss = 0
            for b in range(n_batch_evalu):
                batch_data = evalu_data[b*batch_size:(b+1)*batch_size]
                hist_data = [d[0] for d in batch_data]
                futu_traj = [d[1] for d in batch_data]
                hist_data = torch.from_numpy(np.array(hist_data)).float().to(use_device)
                futu_traj = torch.from_numpy(np.array(futu_traj)).float().to(use_device)

                with torch.no_grad():
                    plan_traj = model.forward(hist_data)
                    loss_ = lossfuc.forward(plan_traj, futu_traj)
                    evalu_loss = evalu_loss + loss_

            evalu_loss /= n_batch_evalu
            
            if stopper.best_score is None or evalu_loss.item() < stopper.best_score:
                torch.save(model.state_dict(), 'aslstm.pt')
            stopper.check(e, evalu_loss)
            if stopper.early_stop:
                print("Early stop at best epoch = %d"%(e))
                break
        
        self.model = model.cpu()

    def test(self, show=False):
        model = self.model.to(use_device)
        dataset = Dataset(use_ds=datci.NGSIM_I80_0500_0515)
        evalu_data = dataset.evalu_data
        model.eval()

        gt_trajs, pl_trajs = [], []
        for data in evalu_data:
            hist_data, futu_traj = data
            hist_data = torch.from_numpy(np.array(hist_data)).float().unsqueeze(0).to(use_device)
            futu_traj = torch.from_numpy(np.array(futu_traj)).float().unsqueeze(0).to(use_device)


            with torch.no_grad():
                plan_traj = model.forward(hist_data)
            
            gt_trajs += [t_.cpu().numpy() for t_ in futu_traj]
            pl_trajs += [t_.cpu().numpy() for t_ in plan_traj]

        gt_trajs = np.array(gt_trajs, float)
        pl_trajs = np.array(pl_trajs, float)
        for k in [10,20,30,40]:
            ee = gt_trajs[k-1,:] - pl_trajs[k-1,:]
            rmse = np.sqrt(np.mean(np.sum(ee**2)))
            print("evaluate at t = %.1f: RMSE = %.2f" % (k*simu_pace, rmse))
        
        if show:
            for data, gt, pl in zip(evalu_data, gt_trajs, pl_trajs):
                hist_data = data[0]
                se, sp, sb, sf, sr = hist_data.transpose((1,0,2))

                plt.figure(1)
                for s_ in [sp, sb, sf, sr]:
                    plt.plot(s_[:,0], s_[:,1], c='tab:blue',marker='o')
                plt.plot(se[:,0], se[:,1], c='tab:red', marker='o')
                plt.plot(gt[:,0], gt[:,1], c='tab:green', marker='o')
                plt.plot(pl[:,0], pl[:,1], c='yellow', marker='*')

                plt.show()

    def train_control(self):
        print('[yellow] train control !!! [/yellow]')
        model = self.model.to(use_device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        stopper = EarlyStopping(patience=50, delta=0.0)
        dataset = Dataset(use_ds=datci.NGSIM_I80_0400_0415, train_ratio=0.8)
        train_data, evalu_data = dataset.train_data, dataset.evalu_data
        n_batch_train = len(train_data) // batch_size if len(train_data) % batch_size == 0 else len(train_data) // batch_size + 1
        n_batch_evalu = len(evalu_data) // batch_size if len(evalu_data) % batch_size == 0 else len(evalu_data) // batch_size + 1

        for e in range(1000):
            random.shuffle(train_data)
            model.train()
            for b in range(n_batch_train):
                batch_data = train_data[b*batch_size:(b+1)*batch_size]
                hist_data = [d[0] for d in batch_data]
                futu_ctrl = [d[2] for d in batch_data]
                hist_data = torch.from_numpy(np.array(hist_data)).float().to(use_device)
                futu_ctrl = torch.from_numpy(np.array(futu_ctrl)).float().to(use_device)

                plan_ctrl = model.forward(hist_data)
                loss_ = lossfuc(plan_ctrl, futu_ctrl)

                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
            
            model.eval()
            evalu_loss = 0
            for b in range(n_batch_evalu):
                batch_data = evalu_data[b*batch_size:(b+1)*batch_size]
                hist_data = [d[0] for d in batch_data]
                futu_ctrl = [d[2] for d in batch_data]
                hist_data = torch.from_numpy(np.array(hist_data)).float().to(use_device)
                futu_ctrl = torch.from_numpy(np.array(futu_ctrl)).float().to(use_device)

                with torch.no_grad():
                    plan_ctrl = model.forward(hist_data)
                    loss_ = lossfuc(plan_ctrl, futu_ctrl)
                    evalu_loss = evalu_loss + loss_

            evalu_loss /= n_batch_evalu
            
            if stopper.best_score is None or evalu_loss.item() < stopper.best_score:
                torch.save(model.state_dict(), 'aslstm.pt')
            stopper.check(e, evalu_loss)
            if stopper.early_stop:
                print("Early stop at best epoch = %d"%(e))
                break
        
        self.model = model.cpu()

    def test_control(self, show=False):
        print('[yellow] test control !!! [/yellow]')
        model = self.model.to(use_device)
        # dataset = Dataset(use_ds=datci.NGSIM_I80_0500_0515)
        dataset = Dataset(use_ds=datci.NGSIM_I80_0515_0530)
        evalu_data = dataset.evalu_data
        model.eval()

        gt_ctrls, pl_ctrls = [], []
        gt_trajs, pl_trajs = [], []
        for data in evalu_data:
            hist_data, futu_traj, futu_ctrl = data
            with torch.no_grad():
                tensor_data = torch.from_numpy(np.array(hist_data)).float().unsqueeze(0).to(use_device)
                plan_ctrl = model.forward(tensor_data)
                plan_ctrl = plan_ctrl.cpu().squeeze(0).numpy()
            
            gt_ctrls.append(futu_ctrl)
            pl_ctrls.append(plan_ctrl)
            gt_trajs.append(get_state_by_control(hist_data[-1,0,:], futu_ctrl, simu_pace)[:,:2])
            pl_trajs.append(get_state_by_control(hist_data[-1,0,:], plan_ctrl, simu_pace)[:,:2])


        gt_ctrls = np.array(gt_ctrls, float)
        pl_ctrls = np.array(pl_ctrls, float)
        gt_trajs = np.array(gt_trajs, float)
        pl_trajs = np.array(pl_trajs, float)
        for k in [10,20,30,40]:
            ee = gt_trajs[:,k-1,:] - pl_trajs[:,k-1,:]
            rmse = np.sqrt(np.sum(np.sum(ee**2, axis=-1)) / len(ee))
            print("evaluate at t = %.1f: RMSE = %.2f" % (k*simu_pace, rmse))
        
        if show:
            for k in range(0, len(evalu_data), 5):
                hist_data = evalu_data[k][0]
                pred_xys = evalu_data[k][1]
                gt_u, gt_s = gt_ctrls[k], gt_trajs[k]
                pl_u, pl_s = pl_ctrls[k], pl_trajs[k]
                se, sp, sb, sf, sr = hist_data.transpose((1,0,2))
                # print(gt_u[::5,0].round(2))
                # print(pl_u[::5,0].round(2))
                # print("------------------------------")
                # print(gt_u[::5,1].round(2))
                # print(pl_u[::5,1].round(2))
                # print("------------------------------")

                plt.figure(1)
                for s_ in [sp, sb, sf, sr]:
                    plt.plot(s_[:,0], s_[:,1], c='gray',marker='o')
                plt.plot(se[:,0], se[:,1], c='red', marker='o')
                plt.plot(pred_xys[:,0], pred_xys[:,1], c='yellow', marker='o')
                plt.plot(gt_s[:,0], gt_s[:,1], c='tab:green', marker='*')
                plt.plot(pl_s[:,0], pl_s[:,1], c='tab:blue', marker='d')
                plt.show()

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> AgentTrajectory:
        self.initialize(EV, lane_from, lane_to)

        if len(self.veh_qlist) < hist_seqlen:
            return None
        
        hist_data = [
            [ [V.x, V.y, V.h, V.v] if V is not None else [0,0,0,0] for V in vehs ] for vehs in self.veh_qlist
        ]
        mask = [
            [ True if V is not None else False for V in vehs ] for vehs in self.veh_qlist
        ]

        hist_data = np.array(hist_data, float)
        x0, y0 = hist_data[-1,0,0], self.lane_to_centery
        hist_data[:,:,0] -= x0 # reset x
        hist_data[:,:,1] -= y0 # reset y
        hist_data[mask == False] = np.zeros(hist_data.shape[-1])

        model = self.model.to(use_device)
        tensor_data = torch.from_numpy(hist_data).reshape(1,hist_seqlen,5,4).float().to(use_device)
        with torch.no_grad():
            plan_ctrl = model.forward(tensor_data)
            plan_ctrl = plan_ctrl.cpu().squeeze(0).numpy()
        plan_traj = get_state_by_control(hist_data[-1,0,:], plan_ctrl, simu_pace)
        plan_traj = np.vstack([hist_data[-1:,0,:], plan_traj])
        plan_traj[:,0] += x0
        plan_traj[:,1] += y0

        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, pred_seqlen+1) * simu_pace + t0,
            states=plan_traj[:,:4],
            inputs=plan_ctrl
        )

        return traj

    def run_replan(self, EV: Agent, lane_from: str, lane_to: str) -> AgentTrajectory:
        return self.run(EV, lane_from, lane_to)
        
    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        self.update_surrounding_vehicles(EV, lane_from, lane_to)

        ########## set ym and yt ##########
        self.lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        self.lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        self.lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        self.lane_to_width = datci.lane.getLaneWidth(self.lane_to)

    def update_surrounding_vehicles(self, EV: Agent, lane_from: str, lane_to: str):
        veh_p, veh_b, veh_f, veh_r = self.get_surrounding_vehicles(EV, lane_from, lane_to)
        if len(self.veh_qlist) == hist_seqlen:
            self.veh_qlist.pop(0)
        self.veh_qlist.append((EV, veh_p, veh_b, veh_f, veh_r))
        self.veh_p, self.veh_b, self.veh_f, self.veh_r = veh_p, veh_b, veh_f, veh_r

    def get_surrounding_vehicles(self, EV:Agent, lane_from:str, lane_to:str) -> tuple[Agent, Agent, Agent, Agent]:
        lane_from_centery = datci.lane.getLane(lane_from).center_line[0][1]
        lane_to_centery = datci.lane.getLane(lane_to).center_line[0][1]

        all_vehicles = datci.getFrameVehicleIds()
        veh_p: Agent = None
        veh_b: Agent = None
        veh_f: Agent = None
        veh_r: Agent = None

        for Vid in all_vehicles:
            if Vid == EV.id:
                continue
            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > EV.x:
                    if veh_p is None or veh_p.x > Vx:
                        veh_p = get_agent_from_daci(Vid)
                else:
                    if veh_b is None or veh_b.x < Vx:
                        veh_b = get_agent_from_daci(Vid)
            elif Vlane == lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > EV.x:
                    if veh_f is None or veh_f.x > Vx:
                        veh_f = get_agent_from_daci(Vid)
                else:
                    if veh_r is None or veh_r.x < Vx:
                        veh_r = get_agent_from_daci(Vid)

        # if veh_p is None:
        #     veh_p = Agent('-1', EV.x+1000, lane_from_centery, 0.0, EV.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        # if veh_b is None:
        #     veh_b = Agent('-2', EV.x-1000, lane_from_centery, 0.0, EV.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        # if veh_f is None:
        #     veh_f = Agent('-3', EV.x+1000, lane_to_centery, 0.0, EV.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        # if veh_r is None:
        #     veh_r = Agent('-4', EV.x-1000, lane_to_centery, 0.0, EV.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        
        return veh_p, veh_b, veh_f, veh_r


if __name__ == '__main__':
    planner = LaneChangePlanner_DLP('aslstm.pt')
    planner.train_control()

    planner = LaneChangePlanner_DLP('aslstm.pt')
    planner.test_control(show=True)