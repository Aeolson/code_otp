import os, sys
import numpy as np
from tqdm import tqdm
import random
from rich import print
from copy import deepcopy
import pickle
import datci
from configs import *
from planner import *
from datasim import DatasetSim

random.seed(10)
np.random.seed(10)


def save_result(sim:DatasetSim, save_dir:str):
    result = {
        'replay_id': sim.replay_id,
        'replay_scenario': sim.replay_scenario,
        'qlist_timestep': sim.qlist_timestep,
        'qlist_ego': sim.qlist_ego,
        'qlist_ego_lane': sim.qlist_ego_lane,
        'qlist_traj': sim.qlist_traj,
        'qlist_risk': sim.qlist_risk,
        'qlist_veh_p': sim.qlist_veh_p,
        'qlist_veh_b': sim.qlist_veh_b,
        'qlist_veh_f': sim.qlist_veh_f,
        'qlist_veh_r': sim.qlist_veh_r,
        'qlist_timecost': sim.qlist_timecost,
        'qlist_planflag': sim.qlist_planflag
    }

    if sim.is_ego_collision:
        flag_ = 'c'
    else:
        if sim.is_plan_success:
            flag_ = 's'
        else:
            flag_ = 'f'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for fg_ in ['c', 'f', 's']:
        fn_ = save_dir + '%d_%s.pkl' % (sim.replay_id, fg_)
        if os.path.isfile(fn_):
            os.remove(fn_)
            print("[blue]Delete the file %s[/blue]" % fn_)

    save_fn = save_dir + '%d_%s.pkl'%(sim.replay_id, flag_)
    with open(save_fn, mode='wb') as fw:
        pickle.dump(result, fw)
        print("[blue]Save the file %s[/blue]" % save_fn)

def run_once(sim:DatasetSim, rp:int, planner:str, show=False, flag_run_sensitivity_analysis=False):

    sim.initilze_scenario(rp)
    if planner == 'dat':
        sim.run_data(show)
    else:
        # save_fn = './results/risk_files_%s/%s/%d_s.pkl' % (use_ds['name'], planner, rp)
        # if os.path.exists(save_fn):
        #     return
        sim.run_plan(planner, show)
    
    if planner == 'otp':
        if flag_run_sensitivity_analysis:
            risk_level = get_risk_level()
            sigma_ax, sigma_ay = get_sigma()
            save_dir = './results/risk_files_%s/%s_@%.1f@%.1f/' % (sim.use_ds['name'], planner, risk_level, sigma_ax)
        else:
            save_dir = './results/risk_files_%s/%s/' % (sim.use_ds['name'], planner)
    else:
        save_dir = './results/risk_files_%s/%s/' % (sim.use_ds['name'], planner)
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_result(sim, save_dir)

    if sim.is_plan_success and not sim.is_ego_collision:
        rf, rr = np.array(sim.qlist_risk).transpose(1,0)
        tc = np.array(sim.qlist_timecost, float)
        pf = np.array(sim.qlist_planflag, int)
        tc = tc[(pf==1) | (pf==2)]
        if len(tc) == 0:
            mean_tc = max_tc = 0
        else:
            mean_tc, max_tc = tc.mean(), tc.max()

        print("[green]Planer %s success !!! risk = (%.2f, %.2f), timecost = (%.3f, %.3f)[/green]" % (planner, rf.max(), rr.max(), mean_tc, max_tc))
    else:
        if sim.is_ego_collision:
            print("[yellow]Planner %s collision !!![/yellow]" % (planner))
        else:
            print("[red]Planner %s fail !!![/red]" % (planner))

def run_validation(show=False):

    for use_ds in [
        datci.NGSIM_I80_0400_0415,
        datci.NGSIM_I80_0500_0515,
        datci.NGSIM_I80_0515_0530
    ]:
        print("Dataset: %s" % (use_ds['name']))
        sim = DatasetSim(use_ds)
        total_num_scenes = len(sim.lane_change_infos)
        for rp in range(total_num_scenes):
            print("[yellow bold]--------------- Scenario %d / %d ---------------" % (rp, total_num_scenes))
            for planner in ['dat', 'otp', 'sbf', 'rap', 'gdp', 'dlp']:
                try:
                    run_once(sim, rp, planner, show)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    pass

def run_ablation(show=False):

    for use_ds in [
        datci.NGSIM_I80_0400_0415,
        datci.NGSIM_I80_0500_0515,
        datci.NGSIM_I80_0515_0530
    ]:
        print("Dataset: %s" % (use_ds['name']))
        sim = DatasetSim(use_ds)
        total_num_scenes = len(sim.lane_change_infos)
        for rp in range(total_num_scenes):
            print("[yellow bold]--------------- Scenario %d / %d ---------------" % (rp, total_num_scenes))
            for planner in ['otp', 'otp_wor', 'otp_nlp']:
                try:
                    run_once(sim, rp, planner, show)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    pass

def run_sensitivity_analysis(show=False):

    for use_ds in [
        datci.NGSIM_I80_0400_0415,
        datci.NGSIM_I80_0500_0515,
        datci.NGSIM_I80_0515_0530
    ]:
        print("Dataset: %s" % (use_ds['name']))
        sim = DatasetSim(use_ds)
        total_num_scenes = len(sim.lane_change_infos)
        for rp in range(total_num_scenes):
            print("[yellow bold]--------------- Scenario %d / %d ---------------" % (rp, total_num_scenes))
            for rs in [0.1]:
                set_risk_level(rs)
                for sx in [0.3,0.6,0.9,1.2]:
                    set_sigma(sx, 0.02)
                    try:
                        run_once(sim, rp, 'otp', show, flag_run_sensitivity_analysis=True)
                    except KeyboardInterrupt:
                        sys.exit()
                    except:
                        pass

if __name__ == '__main__':  

    run_validation()
    # run_ablation()
    # run_sensitivity_analysis()
