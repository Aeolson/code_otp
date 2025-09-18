import os, sys
import datci
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'  #['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom'] 
# matplotlib.rcParams['text.usetex'] = True

font1_size = 10
font1_name = 'Times New Roman'
font1 = {'family':font1_name, 'weight':'normal', 'size':font1_size}

from datasim import DatasetSim
from rich import print
from configs import *

color_dat = '#1E90FF' # blue
color_sbf = '#FFC125' # yellow
color_dlp = '#66CD00' # green
color_otp = '#FF4500' # red

label_dat = r'$\rm Data$'
label_sbf = r'$\rm SBF$'
label_dlp = r'$\rm DLP$'
label_otp = r'$\rm OTP$'

# [4, 22, 24, 41, 46, 49, 58, 63, 93, 141, 170, 187, 198, 201, 202, 218, 244, 248, 251, 259, 287, 307, 333, 335, 340]
# L: [335, 307, 287, 259, 251, 248, 202, 201]
# R: [333, ]
# eva_ids = [333] # L: 287;248;, R: 333
# eva_ds = datci.NGSIM_I80_0400_0415

# # left change:
# eva_ids = [143] 
# eva_ds = datci.NGSIM_I80_0515_0530
eva_ids = [248] 
eva_ds = datci.NGSIM_I80_0400_0415

# right change:
# eva_ids = [333] 
# eva_ds = datci.NGSIM_I80_0400_0415

def filter_risk(ds):
    risk_list = []
    lane_list = []
    save_dir_dat = './results/risk_files_%s/dat' % ds['name']
    save_dir_otp = './results/risk_files_%s/otp' % ds['name']
    save_dir_sbf = './results/risk_files_%s/sbf' % ds['name']
    save_dir_dlp = './results/risk_files_%s/dlp' % ds['name']

    fns = os.listdir(save_dir_dat)
    ids_dat = []
    for f_ in fns:
        id_, flag_ = f_[:-4].split('_')
        if flag_ == 's':
            ids_dat.append(int(id_))
    ids_dat = np.array(ids_dat)

    fns = os.listdir(save_dir_otp)
    ids_otp = []
    for f_ in fns:
        id_, flag_ = f_[:-4].split('_')
        if flag_ == 's':
            ids_otp.append(int(id_))
    ids_otp = np.array(ids_otp)

    fns = os.listdir(save_dir_sbf)
    ids_sbf = []
    for f_ in fns:
        id_, flag_ = f_[:-4].split('_')
        if flag_ == 's':
            ids_sbf.append(int(id_))
    ids_sbf = np.array(ids_sbf)

    fns = os.listdir(save_dir_dlp)
    ids_dlp = []
    for f_ in fns:
        id_, flag_ = f_[:-4].split('_')
        if flag_ == 's':
            ids_dlp.append(int(id_))
    ids_dlp = np.array(ids_dlp)

    for id in ids_dat:
        if id not in ids_otp or id not in ids_sbf or id not in ids_dlp:
            continue

        with open('%s/%d_s.pkl'%(save_dir_dat, id), 'rb') as fr:
            res_dat = pickle.load(fr)
            r = np.array(res_dat['qlist_risk'], float)
            risk_dat = [r.mean(), r.max(), r[:,0].max(), r[:,1].max()]
            laneid = res_dat['qlist_ego_lane']
            from_lane = int(laneid[0].split('_')[-1])
            to_lane = int(laneid[-1].split('_')[-1])
            dir_lc = 'L' if to_lane >= from_lane else 'R'
        with open('%s/%d_s.pkl'%(save_dir_otp, id), 'rb') as fr:
            res_otp = pickle.load(fr)
            r = np.array(res_otp['qlist_risk'], float)
            risk_otp = [r.mean(), r.max(), r[:,0].max(), r[:,1].max()]
        with open('%s/%d_s.pkl'%(save_dir_sbf, id), 'rb') as fr:
            res_sbf = pickle.load(fr)
            r = np.array(res_sbf['qlist_risk'], float)
            risk_sbf = [r.mean(), r.max(), r[:,0].max(), r[:,1].max()]
        with open('%s/%d_s.pkl'%(save_dir_dlp, id), 'rb') as fr:
            res_dlp = pickle.load(fr)
            r = np.array(res_dlp['qlist_risk'], float)
            risk_dlp = [r.mean(), r.max(), r[:,0].max(), r[:,1].max()]
        
        # lane_from = int(res_dat['replay_scenario']['lane_from'].split('_')[-1])
        # lane_to = int(res_dat['replay_scenario']['lane_to'].split('_')[-1])
        # if lane_to > lane_from:
        #     continue
        
        # if (
        #     risk_otp[0] < np.min([risk_dat[0], risk_sbf[0], risk_dlp[0]]) and
        #     risk_otp[1] < np.min([risk_dat[1], risk_sbf[1], risk_dlp[1]]) and
        #     risk_otp[2] < np.min([risk_dat[2], risk_sbf[2], risk_dlp[2]])
        # ):
        #     risk_list.append([id, res_dat, res_otp, res_sbf, res_dlp])
        #     lane_list.append([from_lane, to_lane, dir_lc])


        
        risk_list.append([id, res_dat, res_otp, res_sbf, res_dlp])
        lane_list.append([from_lane, to_lane, dir_lc])
    
    print([r[0] for r in risk_list])
    print([l[-1] for l in lane_list])
    return risk_list

def replay_scenario(ds):
    risk_list = filter_risk(ds)
    sim = DatasetSim(ds)

    for info in risk_list:
        id, res_dat, res_otp, res_sbf, res_dlp = info
        if id not in eva_ids:
            continue

        print("%s : replay_id = %d" % (ds['name'], id))
        print("risk_avg: dat = %.2f, otp = %.2f, sbf = %.2f, dlp = %.2f" % tuple(np.array(r['qlist_risk']).mean() for r in [res_dat, res_otp, res_sbf, res_dlp]))
        print("risk_max: dat = %.2f, otp = %.2f, sbf = %.2f, dlp = %.2f" % tuple(np.array(r['qlist_risk']).max()  for r in [res_dat, res_otp, res_sbf, res_dlp]))
        print("risk_mfr: dat = %.2f, otp = %.2f, sbf = %.2f, dlp = %.2f" % tuple(np.array(r['qlist_risk'])[:,0].max() for r in [res_dat, res_otp, res_sbf, res_dlp]))
        sim.initilze_scenario(id)

        print("[green]------------ Replay DAT --------------[/green]")
        sim.run_replay(res_dat['qlist_timestep'], res_dat['qlist_ego'], res_dat['qlist_traj'], show=True, traj_type='dat')
        print("[green]------------ Replay OTP --------------[/green]")
        sim.run_replay(res_otp['qlist_timestep'], res_otp['qlist_ego'], res_otp['qlist_traj'], show=True, traj_type='otp')
        print("[green]------------ Replay sbf --------------[/green]")
        sim.run_replay(res_sbf['qlist_timestep'], res_sbf['qlist_ego'], res_sbf['qlist_traj'], show=True, traj_type='sbf')
        print("[green]------------ Replay DLP --------------[/green]")
        sim.run_replay(res_dlp['qlist_timestep'], res_dlp['qlist_ego'], res_dlp['qlist_traj'], show=True, traj_type='dlp')

def replay_risk(ds):
    risk_list = filter_risk(ds)
    sim = DatasetSim(ds)

    for info in risk_list:
        id, res_dat, res_otp, res_sbf, res_dlp = info
        if id not in eva_ids:
            continue
        
        ts = np.array(res_dat['qlist_timestep']) * 0.1
        risk_dat = np.array(res_dat['qlist_risk'])#[::2,:]
        risk_otp = np.array(res_otp['qlist_risk'])#[::2,:]
        risk_sbf = np.array(res_sbf['qlist_risk'])
        risk_dlp = np.array(res_dlp['qlist_risk'])

        plt.figure(1, figsize=(4,1.5), dpi=300)
        plt.plot(ts, risk_dat.max(-1), c=color_dat, ls='-',   lw=1.5, label=label_dat)
        plt.plot(ts, risk_sbf.max(-1), c=color_sbf, ls='--',  lw=1.5, label=label_sbf)
        plt.plot(ts, risk_dlp.max(-1), c=color_dlp, ls='-.',  lw=1.5, label=label_dlp)
        plt.plot(ts, risk_otp.max(-1), c=color_otp, ls=':',   lw=1.5, label=label_otp)
        plt.legend(prop=font1, loc='upper left', bbox_to_anchor=(1.0,1.0), ncol=1, columnspacing=0.7, labelspacing=0.7, handletextpad=0.15, handlelength=1.2, framealpha=0.7)
        plt.xlim(ts[0], ts[-1])
        plt.xticks(np.arange(np.ceil(ts[0]), np.floor(ts[-1])+1).tolist())
        plt.ylim(-0.1, 1.1)
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        plt.grid(True, color='tab:red', linestyle=':', linewidth=0.5, alpha=0.4)
        plt.xlabel(r'$\rm time ~ [s]$', labelpad=0.8)
        plt.ylabel(r'$\rm Risk$', labelpad=8.0)

        labs = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [l_.set_fontname(font1_name) for l_ in labs]
        plt.tight_layout()
        plt.subplots_adjust(top=0.95,bottom=0.26,left=0.13,right=0.82,hspace=0,wspace=0)
        plt.savefig("fig_case_%d_risk.png"%id,dpi=600)


        ego_dat = np.array([(ev.x, ev.y, ev.h, ev.v, ev.a, ev.w) for ev in res_dat['qlist_ego']])
        ego_otp = np.array([(ev.x, ev.y, ev.h, ev.v, ev.a, ev.w) for ev in res_otp['qlist_ego']])
        ego_sbf = np.array([(ev.x, ev.y, ev.h, ev.v, ev.a, ev.w) for ev in res_sbf['qlist_ego']])
        ego_dlp = np.array([(ev.x, ev.y, ev.h, ev.v, ev.a, ev.w) for ev in res_dlp['qlist_ego']])

        plt.figure(2, figsize=(4,1.5), dpi=300)
        k=4
        plt.fill_between([ts[0], ts[-1]], [a_min, a_min], [a_max, a_max], color='gray', alpha=0.3)
        plt.plot(ts, ego_dat[:,k], c=color_dat, ls='-',  lw=1.5, label=label_dat)
        plt.plot(ts, ego_sbf[:,k], c=color_sbf, ls='--', lw=1.5, label=label_sbf)
        plt.plot(ts, ego_dlp[:,k], c=color_dlp, ls='-.', lw=1.5, label=label_dlp)
        plt.plot(ts, ego_otp[:,k], c=color_otp, ls=':',  lw=1.5, label=label_otp)
        plt.legend(prop=font1, loc='upper left', bbox_to_anchor=(1.0,1.0), ncol=1, columnspacing=0.7, labelspacing=0.7, handletextpad=0.15, handlelength=1.2, framealpha=0.7)
        plt.xlim(ts[0], ts[-1])
        plt.xticks(np.arange(np.ceil(ts[0]), np.floor(ts[-1])+1).tolist())
        plt.ylim(a_min-0.5, a_max+0.5)
        plt.yticks([-4,-2,0,2])
        plt.grid(True, color='tab:red', linestyle=':', linewidth=0.5, alpha=0.4)
        plt.xlabel(r'$\rm time ~ [s]$', labelpad=0.8)
        plt.ylabel(r'$\rm Acceleration ~ [m/s^2]$', labelpad=8.0)

        labs = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [l_.set_fontname(font1_name) for l_ in labs]
        plt.tight_layout()
        plt.subplots_adjust(top=0.95,bottom=0.26,left=0.13,right=0.82,hspace=0,wspace=0)
        plt.savefig("fig_case_%d_acc.png"%id,dpi=600)

        
        plt.figure(3, figsize=(4,1.5), dpi=300)
        k=5
        plt.fill_between([ts[0], ts[-1]], [np.tan(d_min), np.tan(d_min)], [np.tan(d_max), np.tan(d_max)], color='gray', alpha=0.3)
        plt.plot(ts, ego_dat[:,k], c=color_dat, ls='-',  lw=1.5, label=label_dat)
        plt.plot(ts, ego_sbf[:,k], c=color_sbf, ls='--', lw=1.5, label=label_sbf)
        plt.plot(ts, ego_dlp[:,k], c=color_dlp, ls='-.', lw=1.5, label=label_dlp)
        plt.plot(ts, ego_otp[:,k], c=color_otp, ls=':',  lw=1.5, label=label_otp)
        plt.legend(prop=font1, loc='upper left', bbox_to_anchor=(1.0,1.0), ncol=1, columnspacing=0.7, labelspacing=0.7, handletextpad=0.15, handlelength=1.2, framealpha=0.7)
        plt.xlim(ts[0], ts[-1])
        plt.xticks(np.arange(np.ceil(ts[0]), np.floor(ts[-1])+1).tolist())
        plt.ylim(d_min-0.1, d_max+0.1)
        plt.yticks([-0.4,-0.2,0.0,0.2,0.4])
        plt.xlim(ts[0], ts[-1])
        plt.grid(True, color='tab:red', linestyle=':', linewidth=0.5, alpha=0.4)
        plt.xlabel(r'$\rm time ~ [s]$', labelpad=0.8)
        plt.ylabel(r'$\delta_f ~ \rm [rad]$', labelpad=0.5)

        labs = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [l_.set_fontname(font1_name) for l_ in labs]
        plt.tight_layout()
        plt.subplots_adjust(top=0.95,bottom=0.26,left=0.13,right=0.82,hspace=0,wspace=0)
        plt.savefig("fig_case_%d_dfw.png"%id,dpi=600)


        plt.show()

if __name__ == '__main__':
    # replay_scenario(eva_ds)
    replay_risk(eva_ds)
    