import torch
import numpy as np
import os
import sys

from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from sklearn.model_selection import train_test_split

import ray
import tracemalloc
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import wandb

pwd = os.getcwd()
os.chdir(pwd)
sys.path.append(pwd)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 定义一个 lambda 函数来逐步增加学习率
def get_loader(data: np.ndarray, cond: np.ndarray, shuffle=True, batch_size=64):
    dataset = TensorDataset(
        torch.from_numpy(data.astype(int)),
        torch.from_numpy(cond.astype(int)).unsqueeze(1),
    )

    dataloader = DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=4
    )
    return dataloader


def npy_to_loader(data_path, cond_path, batch_size, valid_size=0.2):
    data = np.load(data_path, allow_pickle=True)
    cond = np.load(cond_path, allow_pickle=True)
    train_data, test_data, train_cond, test_cond = train_test_split(
        data, cond, test_size=valid_size
    )
    train_loader = get_loader(
        train_data, train_cond, batch_size=batch_size, shuffle=True
    )
    test_loader = get_loader(test_data, test_cond, batch_size=batch_size, shuffle=True)
    return (train_loader, test_loader)


def train(config):
    run = wandb.init(project="sc-maskgit", config=config)
    set_seed()
    transformer = MaskGitTransformer(
        num_tokens=51,  # must be same as codebook size above
        seq_len=1200,  # must be equivalent to fmap_size ** 2 in vae
        dim=512,  # model dimension
        depth=3,  # depth
        dim_head=64,  # attention head dimension
        heads=8,  # attention heads,
        ff_mult=4,  # feedforward expansion factor
        t5_name="t5-small",  # name of your T5, 如果没有 text 可以不用
    )

    base_maskgit = MaskGit(
        vae=None,  # vqgan vae
        transformer=transformer,  # transformer
        image_size=256,  # image size, 如果不做图像超分可以不用
        cond_drop_prob=0.25,  # conditional dropout, for classifier free guidance
    ).cuda()

    train_loader, valid_loader = npy_to_loader(
        "/home/lijiahao/workbench/sc-maskgit/data/data_bins.npy", "/home/lijiahao/workbench/sc-maskgit/data/condition.npy", batch_size=64
    )

    optimizer = torch.optim.Adam(
        base_maskgit.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    # 定义一个 lambda 函数来逐步增加学习率
    def lr_lambda(step):
        if step < config['lr_milestone']:
            return step / config['lr_milestone']  # 学习率线性增加
        else:
            return 1.0  # 达到目标学习率后保持不变

    # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1)
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_step'] - config['lr_milestone']
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, milestones=[config['lr_milestone']], schedulers=[scheduler1, scheduler2]
    )

    epoch = np.ceil(config['max_step'] / len(train_loader))
    # tq = tqdm(range(int(epoch)))
    step = 0
    for _ in range(int(epoch)):
        for images, texts in train_loader:
            images = images.cuda()
            texts = texts.cuda()
            loss = base_maskgit(images, texts=texts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            run.log(
                {
                    "BCE loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "train_step": step,
                }
            )
            session.report({"loss": loss.item()})
            step += 1
            if step >= config['max_step']:
                break
        with torch.no_grad():
            for images, texts in valid_loader:
                images = images.cuda()
                texts = texts.cuda()
                loss = base_maskgit(images, texts=texts)
                run.log({"Valid BCE loss": loss.item()})

    torch.save(base_maskgit.state_dict(), "maskgit.pt")


def ray_tune():
    ray.init(num_cpus=40, num_gpus=1)
    tracemalloc.start()
    step_range = [600]
    search_space = {
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "max_step": tune.choice(step_range),
        "lr_milestone": tune.choice([100]),
        # "enc_dim": tune.choice(range(20, 25)),
    }

    scheduler = ASHAScheduler(max_t=max(step_range), grace_period=10, reduction_factor=4)

    optuna_search = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train), resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=50,
            metric="loss",
            mode="min",
        ),
        param_space=search_space,
    )

    results = tuner.fit()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    ray_tune()
