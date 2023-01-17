'''
Date: 2021-11-28 13:04:02
LastEditors: Liu Yahui
LastEditTime: 2022-05-18 12:24:51
'''
# Reference: https://github.com/tiangexiang/CurveNet


import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F

from deepspeed.profiling.flops_profiler import get_model_profile

import shutil
import os
import subprocess


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss



def critical_feature_sample(feature, npoint):
    device = feature.device
    B, S, D = feature.shape 
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    
    for i in range(B):
        fp_per = feature[i, :, :].detach().cpu().numpy()   # [S, D]
        drop_idx = np.arange(S)  # [S]
        k = npoint
        sample_idx = []
        while True:
            idx = fp_per.argmax(0)  # [D]
            idx = drop_idx[idx]
            uidx = np.unique(idx)
            len_uidx = len(uidx)  
            if  len_uidx > k:
                values, _ = zip(*Counter(idx).most_common(k))
                sample_idx.extend(list(values))
                break

            elif len_uidx < k:
                sample_idx.extend(list(uidx))
                l = fp_per.shape[0]
                all_idx = np.arange(l)
                drop_idx = np.array(list(set(all_idx) - set(uidx)))
                fp_per = fp_per[drop_idx, :]
                k = k - len_uidx

            else:
                sample_idx.extend(list(uidx))
                break

        centroids[i, :] = torch.Tensor(sample_idx)

    return centroids
        

def profile_model(model, args):
    # model.eval()  # model eval
    flops, macs, params = get_model_profile(
        model=model,
        args=args,
        print_profile=False,  # prints the model graph with the measured profile attached to each module
        detailed=False,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return flops, macs, params



class WandbUrls:
    def __init__(self, url):

        hash = url.split("/")[-2]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = "https://app.wandb.ai/{}/{}/runs/{}/logs".format(entity, project, hash)
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(entity, project, hash)
        self.overview_url = "https://app.wandb.ai/{}/{}/runs/{}/overview".format(entity, project, hash)
        self.config_url = "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
            entity, project, hash
        )
        self.overrides_url = "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(entity, project, hash)

    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"
        return msg


class Wandb:
    IS_ACTIVE = False

    @staticmethod
    def set_urls_to_model(model, url):
        wandb_urls = WandbUrls(url)
        model.wandb = wandb_urls

    @staticmethod
    def _set_to_wandb_args(wandb_args, cfg, name):
        var = getattr(cfg.wandb, name, None)
        if var:
            wandb_args[name] = var

    @staticmethod
    def launch(cfg, launch: bool):
        if launch:
            import wandb

            Wandb.IS_ACTIVE = True

            wandb_args = {}
            wandb_args["resume"] = "allow"
            Wandb._set_to_wandb_args(wandb_args, cfg, "tags")
            Wandb._set_to_wandb_args(wandb_args, cfg, "project")
            Wandb._set_to_wandb_args(wandb_args, cfg, "name")
            Wandb._set_to_wandb_args(wandb_args, cfg, "entity")
            Wandb._set_to_wandb_args(wandb_args, cfg, "notes")
            Wandb._set_to_wandb_args(wandb_args, cfg, "config")
            Wandb._set_to_wandb_args(wandb_args, cfg, "id")

            try:
                commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                gitdiff = subprocess.check_output(["git", "diff", "--", "':!notebooks'"]).decode()
            except:
                commit_sha = "n/a"
                gitdiff = ""

            config = wandb_args.get("config", {})
            wandb_args["config"] = {
                **config,
                "run_path": os.getcwd(),
                "commit": commit_sha,
                "gitdiff": gitdiff
            }
            wandb.init(**wandb_args, sync_tensorboard=True)
            wandb.save(os.path.join(os.getcwd(), cfg.cfg_path))

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError("wandb is inactive, please launch first.")
        import wandb

        filename = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(wandb.run.dir, filename))
