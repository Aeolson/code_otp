import os, sys
import datci
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为 中文宋体 + 英文Times New Roman + latex
from matplotlib import font_manager
from matplotlib import rcParams
font_path = "./times+simsun.ttf" # 加载字体
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = 'sans-serif' # 使用字体中的无衬线体
rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
rcParams['font.size'] = 12 # 设置字体大小
rcParams['axes.unicode_minus'] = False # 使坐标轴刻度标签正常显示正负号
rcParams['mathtext.fontset'] = 'cm' # latex 公式字体

from utils.simple_agent import Agent, calc_agent_risk



all_ds = [
    datci.NGSIM_I80_0400_0415,
    datci.NGSIM_I80_0500_0515, 
    datci.NGSIM_I80_0515_0530,
]


test_planner = [
    'dat',

    'otp_@0.1@0.3',
    'otp_@0.1@0.6',
    'otp_@0.1@0.9',
    'otp_@0.1@1.2',

    'otp_@0.2@0.3',
    'otp_@0.2@0.6',
    'otp_@0.2@0.9',
    'otp_@0.2@1.2',

    'otp_@0.3@0.3',
    'otp_@0.3@0.6',
    'otp_@0.3@0.9',
    'otp_@0.3@1.2',

    'otp_@0.4@0.3',
    'otp_@0.4@0.6',
    'otp_@0.4@0.9',
    'otp_@0.4@1.2',

    'otp_@0.5@0.3',
    'otp_@0.5@0.6',
    'otp_@0.5@0.9',
    'otp_@0.5@1.2',
]


def load_all_data():
    all_data = {}

    for ds in all_ds:
        ds_data = {}

        print("------------------------------------------------------------------------------")
        print("Dataset : %s" % ds['name'])
        for planner in test_planner:
            ds_data[planner] = load_data(ds, planner)
        
        print("------------------------------------------------------------------------------")
        
        all_data[ds['name']] = ds_data

    # print success rate
    for pl in test_planner:
        cnt_s, cnt_f, cnt_c = 0, 0, 0
        ts_avg, ts_max = [], []
        r_avg, r_max, r_max_f, r_max_r = [], [], [], []
        for ds in all_ds:
            data = all_data[ds['name']][pl]
            cnt_s += len(data['succ_data'])
            cnt_f += len(data['fail_data'])
            cnt_c += len(data['coll_data'])

            ts_avg += [d['plan_time']['avg'] for d in data['succ_data']]
            ts_max += [d['plan_time']['max'] for d in data['succ_data']]

            # r_avg   += [d['risks']['avg']   for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            # r_max   += [d['risks']['max']   for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            # r_max_f += [d['risks']['max_f'] for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            # r_max_r += [d['risks']['max_r'] for d in data['succ_data'] + data['fail_data'] + data['coll_data']]

            r_avg += [np.mean([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max += [np.max([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_f += [np.max([d['risks_km'][v] for v in ['p', 'f']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_r += [np.max([d['risks_km'][v] for v in ['b', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]

        cnt_total = cnt_s + cnt_f + cnt_c
        print("%-12s : rate = & %5.1f & %5.1f & %5.1f, risk = & %2.3f & %2.3f & %2.3f & %2.3f, time = & %2.3f & %2.3f"% (
            pl, 
            cnt_s / cnt_total * 100, cnt_f / cnt_total * 100, cnt_c / cnt_total * 100, 
            np.mean(r_avg), np.mean(r_max), np.mean(r_max_f), np.mean(r_max_r), 
            np.mean(ts_avg), np.mean(ts_max)
        ))
    
    return all_data

def load_data(ds:dict, planner:str):
    succ_data, fail_data, coll_data = [], [], []

    load_dir = './results/risk_files_%s/%s/' % (ds['name'], planner)
    fns = os.listdir(load_dir)

    for f_ in fns:

        with open('%s%s'%(load_dir, f_), 'rb') as fr:
            ditem = process_data(pickle.load(fr))
    
        t_ = f_[:-4].split('_')[1]
        if t_ == 's':
            succ_data.append(ditem)
        elif t_ == 'f':
            fail_data.append(ditem)
        elif t_ == 'c':
            coll_data.append(ditem)
    
    total_num = len(succ_data) + len(fail_data) + len(coll_data)
    succ_rate = len(succ_data) / total_num
    fail_rate = len(fail_data) / total_num
    coll_rate = len(coll_data) / total_num

    r_avg = np.mean([np.mean([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in succ_data + fail_data + coll_data])
    r_max = np.mean([np.max([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in succ_data + fail_data + coll_data])
    r_max_f = np.mean([np.max([d['risks_km'][v] for v in ['p', 'f']]) for d in succ_data + fail_data + coll_data])
    r_max_r = np.mean([np.max([d['risks_km'][v] for v in ['b', 'r']]) for d in succ_data + fail_data + coll_data])


    print("planner = {:<12s} : rate = & {:>5.1f} & {:>5.1f} & {:>5.1f} ; risk = & {:>2.3f} & {:>2.3f} & {:>2.3f} & {:>2.3f}".format(
        planner, succ_rate*100, fail_rate*100, coll_rate*100, r_avg, r_max, r_max_f, r_max_r
    ))

    return {
        'succ_data': succ_data,
        'fail_data': fail_data,
        'coll_data': coll_data
    }

def process_data(data:dict):
    replay_id = data['replay_id']
    replay_scenario = data['replay_scenario']
    qlist_timestep = data['qlist_timestep']
    qlist_ego = data['qlist_ego']
    qlist_ego_lane = data['qlist_ego_lane']
    qlist_traj = data['qlist_traj']
    qlist_risk = data['qlist_risk']
    qlist_veh_p = data['qlist_veh_p']
    qlist_veh_b = data['qlist_veh_b']
    qlist_veh_f = data['qlist_veh_f']
    qlist_veh_r = data['qlist_veh_r']
    qlist_timecost = data['qlist_timecost']
    qlist_planflag = data['qlist_planflag']

    r = np.array(qlist_risk, float) # (N, 2)
    risks = ( r.mean(), r.max(), r[:,0].max(), r[:,1].max() )

    tc = np.array(qlist_timecost, float)
    pf = np.array(qlist_planflag, int)
    tc = tc[pf == 1]
    if len(tc) == 0:
        plan_time = (0.0, 0.0)
    else:
        plan_time = (np.mean(tc), np.max(tc))
    
    # merging instant distance
    risk_km = [0.0, 0.0, 0.0, 0.0]
    
    k_ = np.argwhere(np.array(qlist_ego_lane) != qlist_ego_lane[0]).reshape(-1)
    if len(k_) > 0:
        km = k_[0]
        veh_e : Agent = qlist_ego[km]
        veh_p : Agent = qlist_veh_p[km]
        veh_b : Agent = qlist_veh_b[km]
        veh_f : Agent = qlist_veh_f[km]
        veh_r : Agent = qlist_veh_r[km]
        
        # risk km
        if veh_p is not None:
            risk_km[0] = calc_agent_risk(veh_e, veh_p)
        if veh_b is not None:
            risk_km[1] = calc_agent_risk(veh_e, veh_b)
        if veh_f is not None:
            risk_km[2] = calc_agent_risk(veh_e, veh_f)
        if veh_r is not None:
            risk_km[3] = calc_agent_risk(veh_e, veh_r)
    
    return {
        'replay_id': replay_id,
        'plan_time': {'avg': plan_time[0], 'max': plan_time[1]},
        'risks': {'avg': risks[0], 'max': risks[1], 'max_f': risks[2], 'max_r': risks[3]},
        'risks_km': {'p': risk_km[0], 'b': risk_km[1], 'f': risk_km[2], 'r': risk_km[3]}
    }

def plot_hist(all_data):

    all_risks = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    all_sigmas = np.array([0.3, 0.6, 0.9, 1.2])
    xx, yy = np.meshgrid(all_risks, all_sigmas)
    zz_succ = np.zeros_like(xx)
    zz_fail = np.zeros_like(xx)
    zz_ravg = np.zeros_like(xx)
    zz_rmax = np.zeros_like(xx)

    for i_, r_ in enumerate(all_risks):
        for j_, s_ in enumerate(all_sigmas):
            pl = 'otp_@%.1f@%.1f'%(r_, s_)

            cnt_s, cnt_f, cnt_c = 0, 0, 0
            ts_avg, ts_max = [], []
            r_avg, r_max, r_max_f, r_max_r = [], [], [], []
            for ds in all_ds:
                data = all_data[ds['name']][pl]
                cnt_s += len(data['succ_data'])
                cnt_f += len(data['fail_data'])
                cnt_c += len(data['coll_data'])

                ts_avg += [d['plan_time']['avg'] for d in data['succ_data']]
                ts_max += [d['plan_time']['max'] for d in data['succ_data']]

                r_avg   += [d['risks']['avg']   for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
                r_max   += [d['risks']['max']   for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            
            zz_succ[j_,i_] = cnt_s / (cnt_s + cnt_f + cnt_c) * 100.0
            zz_fail[j_,i_] = cnt_f / (cnt_s + cnt_f + cnt_c) * 100.0
            zz_ravg[j_,i_] = np.mean(r_avg)
            zz_rmax[j_,i_] = np.mean(r_max)
    
    

    fig = plt.figure(1, (2.82, 2.5), dpi=200)
    ax : Axes3D = fig.add_subplot(111, projection='3d')
    print(zz_succ.min(), zz_succ.max())

    x_, y_, z_ = xx.ravel(), yy.ravel(), zz_succ.ravel()
    colors = plt.cm.viridis((z_ - z_.min()) / (z_.max() - z_.min()))
    ax.bar3d(x_-0.05, y_-0.15, np.zeros_like(z_), 0.1, 0.3, z_, shade=True, color=colors, alpha=0.7)
    ax.set_xlabel(r'$\Delta_r$')
    ax.set_xticks(all_risks)
    ax.set_ylabel(r'$\mathbb{E}[\omega^{x2}]$')
    ax.set_yticks(all_sigmas)
    ax.set_yticklabels([r'$0.3^2$', r'$0.6^2$', r'$0.9^2$', r'$1.2^2$'])
    ax.set_zlabel(r'$D_{\rm succ}$')
    ax.set_zticks([0, 20, 40, 60, 80])
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0.135, right=0.78, top=1.0)
    plt.savefig("fig_hist3d_succ.png",dpi=600)

    fig = plt.figure(2, (2.82, 2.5), dpi=200)
    ax : Axes3D = fig.add_subplot(111, projection='3d')
    print(zz_fail.min(), zz_fail.max())

    x_, y_, z_ = xx.ravel(), yy.ravel(), zz_fail.ravel()
    colors = plt.cm.viridis((z_ - z_.min()) / (z_.max() - z_.min()))
    ax.bar3d(x_-0.05, y_-0.15, np.zeros_like(z_), 0.1, 0.3, z_, shade=True, color=colors, alpha=0.7)
    ax.set_xlabel(r'$\Delta_r$')
    ax.set_xticks(all_risks)
    ax.set_ylabel(r'$\mathbb{E}[\omega^{x2}]$')
    ax.set_yticks(all_sigmas)
    ax.set_yticklabels([r'$0.3^2$', r'$0.6^2$', r'$0.9^2$', r'$1.2^2$'])
    ax.set_zlabel(r'$D_{\rm fail}$')
    ax.set_zticks([0, 20, 40, 60, 80])
    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0.135, right=0.78, top=1.0)
    plt.savefig("fig_hist3d_fail.png",dpi=600)

    fig = plt.figure(3, (2.82, 2.5), dpi=200)
    ax : Axes3D = fig.add_subplot(111, projection='3d')
    print(zz_ravg.min(), zz_ravg.max())

    x_, y_, z_ = xx.ravel(), yy.ravel(), zz_ravg.ravel()
    colors = plt.cm.viridis_r((z_ - z_.min()) / (z_.max() - z_.min()))
    z_lb = np.floor(z_.min() / 0.02) * 0.02
    z_ub = np.ceil(z_.max() / 0.02) * 0.02
    ax.bar3d(x_-0.05, y_-0.15, z_lb*np.ones_like(z_), 0.1, 0.3, z_-z_lb, shade=True, color=colors, alpha=0.7)
    ax.set_xlabel(r'$\Delta_r$')
    ax.set_xticks(all_risks)
    ax.set_ylabel(r'$\mathbb{E}[\omega^{x2}]$')
    ax.set_yticks(all_sigmas)
    ax.set_yticklabels([r'$0.3^2$', r'$0.6^2$', r'$0.9^2$', r'$1.2^2$'])
    ax.set_zlabel(r'$R_{\rm avg}$')
    ax.set_zlim([z_lb, z_ub])
    
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0.135, right=0.78, top=1.0)
    plt.savefig("fig_hist3d_ravg.png",dpi=600)

    fig = plt.figure(4, (2.82, 2.5), dpi=200)
    ax : Axes3D = fig.add_subplot(111, projection='3d')
    print(zz_rmax.min(), zz_rmax.max())

    x_, y_, z_ = xx.ravel(), yy.ravel(), zz_rmax.ravel()
    colors = plt.cm.viridis_r((z_ - z_.min()) / (z_.max() - z_.min()))
    z_lb = np.floor(z_.min() / 0.02) * 0.02
    z_ub = np.ceil(z_.max() / 0.02) * 0.02
    ax.bar3d(x_-0.05, y_-0.15, z_lb*np.ones_like(z_), 0.1, 0.3, z_-z_lb, shade=True, color=colors, alpha=0.7)
    ax.set_xlabel(r'$\Delta_r$')
    ax.set_xticks(all_risks)
    ax.set_ylabel(r'$\mathbb{E}[\omega^{x2}]$')
    ax.set_yticks(all_sigmas)
    ax.set_yticklabels([r'$0.3^2$', r'$0.6^2$', r'$0.9^2$', r'$1.2^2$'])
    ax.set_zlabel(r'$R_{\rm max}$')
    ax.set_zlim([z_lb, z_ub])
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0.135, right=0.78, top=1.0)
    plt.savefig("fig_hist3d_rmax.png",dpi=600)


    plt.show()


if __name__ =='__main__':

    all_data = load_all_data()

    plot_hist(all_data)


