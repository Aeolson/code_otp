import os, sys
import datci
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns

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
    'fsm',
    'rap',
    'gdr',
    'dlp',
    'otp',
    'otp_nlp',
    'otp_wor',
    'otp_wor_nlp',
]

plot_infos = {
    'dat' : {'color': '#56B4E9', 'label': 'Data'},
    'fsm' : {'color': '#E69F00', 'label': 'SBF'},
    'rap' : {'color': '#009E73', 'label': 'RAP'},
    'gdr' : {'color': '#D55E00', 'label': 'GDP'},
    'dlp' : {'color': '#CC79A7', 'label': 'DLP'},
    'otp' : {'color': '#0072B2', 'label': 'OTP'},
}


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
    print("------------------------------------------------------------------------------")
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

            r_avg += [np.mean([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max += [np.max([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_f += [np.max([d['risks_km'][v] for v in ['p', 'f']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_r += [np.max([d['risks_km'][v] for v in ['b', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]

        cnt_total = cnt_s + cnt_f + cnt_c
        print("%-12s : rate = & %5.1f & %5.1f & %5.1f, risk = & %2.3f & %2.3f & %2.3f & %2.3f, time = & %4d & %4d"% (
            pl, 
            cnt_s / cnt_total * 100, cnt_f / cnt_total * 100, cnt_c / cnt_total * 100, 
            np.mean(r_avg), np.mean(r_max), np.mean(r_max_f), np.mean(r_max_r), 
            np.mean(ts_avg)*1000, np.mean(ts_max)*1000
        ))
    
    print("------------------------------------------------------------------------------")
    for pl in test_planner:
        cnt_s, cnt_f, cnt_c = 0, 0, 0
        cnt_sp, cnt_fp, cnt_cp = 0, 0, 0
        cnt_total = 0
        cnt_do_plan = 0
        for ds in all_ds:
            data = all_data[ds['name']][pl]

            cnt_s += len(data['succ_data'])
            for d_ in data['succ_data']:
                if d_['do_plan']:
                    cnt_sp += 1

            cnt_f += len(data['fail_data'])
            for d_ in data['fail_data']:
                if d_['do_plan']:
                    cnt_fp += 1

            cnt_c += len(data['coll_data'])
            for d_ in data['coll_data']:
                if d_['do_plan']:
                    cnt_cp += 1
            
        cnt_total = cnt_s + cnt_f + cnt_c
        cnt_do_plan = cnt_sp + cnt_fp + cnt_cp
                        
        print("%-12s : do_plan = & %d / %d / %4.4f & %d / %d / %4.4f & %d / %d / %4.4f & %d / %d / %4.4f" % (
            pl, 
            cnt_sp, cnt_s, cnt_sp / (cnt_s + 1e-8),
            cnt_fp, cnt_f, cnt_fp / (cnt_f + 1e-8),
            cnt_cp, cnt_c, cnt_cp / (cnt_c + 1e-8),
            cnt_do_plan, cnt_total, cnt_do_plan / (cnt_total + 1e-8)
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

    t_avg = np.mean([d['plan_time']['avg'] for d in succ_data])
    t_max = np.mean([d['plan_time']['max'] for d in succ_data])

    # r_avg = np.mean([d['risks']['avg'] for d in succ_data + fail_data + coll_data])
    # r_max = np.mean([d['risks']['max'] for d in succ_data + fail_data + coll_data])
    # r_max_f = np.mean([d['risks']['max_f'] for d in succ_data + fail_data + coll_data])
    # r_max_r = np.mean([d['risks']['max_r'] for d in succ_data + fail_data + coll_data])

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
        

    if np.count_nonzero(pf>1) > 0:
        do_plan = True
    else:
        do_plan = False
    
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
        'risks_km': {'p': risk_km[0], 'b': risk_km[1], 'f': risk_km[2], 'r': risk_km[3]},
        'do_plan': do_plan
    }

def plot_statistic():
    all_data = {}

    for ds in all_ds:
        ds_data = {}

        print("------------------------------------------------------------------------------")
        print("Dataset : %s" % ds['name'])
        for planner in plot_infos.keys():
            ds_data[planner] = load_data(ds, planner)
        
        print("------------------------------------------------------------------------------")
        
        all_data[ds['name']] = ds_data
    
    risk_dict = {}
    for pl in plot_infos.keys():
        cnt_s, cnt_f, cnt_c = 0, 0, 0
        r_avg, r_max, r_max_f, r_max_r = [], [], [], []
        for ds in all_ds:
            data = all_data[ds['name']][pl]
            cnt_s += len(data['succ_data'])
            cnt_f += len(data['fail_data'])
            cnt_c += len(data['coll_data'])

            r_avg += [np.mean([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max += [np.max([d['risks_km'][v] for v in ['p', 'b', 'f', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_f += [np.max([d['risks_km'][v] for v in ['p', 'f']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]
            r_max_r += [np.max([d['risks_km'][v] for v in ['b', 'r']]) for d in data['succ_data'] + data['fail_data'] + data['coll_data']]

        risk_dict[pl] = {
            'r_avg': np.array(r_avg),
            'r_max': np.array(r_max),
            'r_mxf': np.array(r_max_f),
            'r_mxr': np.array(r_max_r)
        }
    
    for pl in plot_infos.keys():
        print(pl, np.mean(risk_dict[pl]['r_avg']), np.mean(risk_dict[pl]['r_max']), np.mean(risk_dict[pl]['r_mxf']), np.mean(risk_dict[pl]['r_mxr']))

    # figure 1: average risk
    fig = plt.figure(1, (4, 1.5), dpi=200)
    ax = fig.add_subplot(111)
    colors_ = [plot_infos[p_]['color'] for p_ in plot_infos.keys()]
    labels_ = [plot_infos[p_]['label'] for p_ in plot_infos.keys()]
    plt.hist(
        [risk_dict[p_]['r_avg'] for p_ in plot_infos.keys()],
        bins=20,
        range=(0.0,1.0),
        color=colors_,
        label=labels_,
        density=True, histtype='bar', stacked=False, alpha=0.8, align='mid'
    )

    plt.xlim(0.0, 0.6)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.xlabel('Average Risk', labelpad=0.8)
    plt.ylabel('Density')

    plt.legend(loc='upper left', bbox_to_anchor=(1.01,1.1), ncol=1, columnspacing=0.7, labelspacing=0.2, handletextpad=0.15, handlelength=1.2, framealpha=0.7)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96,bottom=0.26,left=0.10,right=0.79,hspace=0,wspace=0)
    plt.savefig("fig_risk_avg.png",dpi=600)

    # figure 2: maximum risk
    fig = plt.figure(2, (4, 1.5), dpi=200)
    ax = fig.add_subplot(111)
    colors_ = [plot_infos[p_]['color'] for p_ in plot_infos.keys()]
    labels_ = [plot_infos[p_]['label'] for p_ in plot_infos.keys()]
    plt.hist(
        [risk_dict[p_]['r_max'] for p_ in plot_infos.keys()],
        bins=20,
        range=(0.0,1.0),
        color=colors_,
        label=labels_,
        density=True, histtype='bar', stacked=False, alpha=0.8, align='mid'
    )

    plt.xlim(0.0, 1.0)
    plt.xlabel('Maximum Risk', labelpad=0.8)
    plt.ylabel('Density')

    plt.legend(loc='upper left', bbox_to_anchor=(1.01,1.1), ncol=1, columnspacing=0.7, labelspacing=0.2, handletextpad=0.15, handlelength=1.2, framealpha=0.7)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96,bottom=0.26,left=0.10,right=0.79,hspace=0,wspace=0)
    plt.savefig("fig_risk_max.png",dpi=600)

    plt.show()

if __name__ =='__main__':

    plot_statistic()


