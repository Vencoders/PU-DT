import os
import shutil
import pathlib
from common.configs import args
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from time import time
import numpy as np
import pointnet2_ops.pointnet2_utils as pointnet2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import utils.pc_util as pc_util
from common.data_loader import PUGAN_Dataset
from common.helper import Logger, adjust_gamma, adjust_learning_rate, save_checkpoint
from common.loss import Loss
from network.model import Model
def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 0    # 仰角
        azim = -45 + 90 * i    # 方位角
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(
                3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir,
                       c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def test(
    model,
    npoints,
    input_val_list,
    gt_val_list,
    centroid_val_list,
    distance_val_list,
    name_list,
):
    start = time()
    val_loss = Loss()
    cd_loss, hd_loss = 0.0, 0.0
    num_sample = len(input_val_list)
    out_folder_pred = os.path.join(args.log_dir, args.out_dir)
    out_folder_coarse = os.path.join(args.log_dir, args.out_dir_coarse)
    if not os.path.exists(out_folder_pred):
        os.makedirs(out_folder_pred)
    if not os.path.exists(out_folder_coarse):
        os.makedirs(out_folder_coarse)
    with torch.no_grad():
        infer_times = []
        for i in range(num_sample):
            N = input_val_list[i].shape[0]
            torch.cuda.empty_cache()
            d = distance_val_list[i].float().to(device)  # 1,1
            c = centroid_val_list[i].float().to(device)  # 1,3
            input_list = input_val_list[i]  # 24 256 3
            input, centorids, furthest_distances = pc_util.normalize_inputs(
                input_list.numpy()
            )
            pred_list = []
            coarse_list = []
            for j in range(N):  # 24
                torch.cuda.empty_cache()
                pts = input[j: (j + 1)]    # 1,256,3
                pts = pts.permute(0, 2, 1).contiguous().float().to(device)
                s = time()
                coarse, pred = model(pts)
                infer_times.append((time() - s))
                pred_list.append(pred.detach())
                coarse_list.append(coarse.detach())
            pred = torch.cat(pred_list)
            coarse = torch.cat(coarse_list)
            pred = pred.permute(0, 2, 1).contiguous()
            coarse = coarse.permute(0, 2, 1).contiguous()
            pred = pred * furthest_distances.to(device) + centorids.to(device)    # 24, 1024, 3
            coarse = coarse * furthest_distances.to(device) + centorids.to(device)
            if args.patch_visualize and args.phase == "test":
                pc_util.patch_visualize(
                    input_list.numpy(), pred.cpu().numpy(), out_folder_pred, name_list[i],
                )
                pc_util.patch_visualize(
                    input_list.numpy(), coarse.cpu().numpy(), out_folder_coarse, name_list[i],
                )
            pred = pred.reshape([1, -1, 3]).float()  
            coarse = coarse.reshape([1, -1, 3]).float()
            pred = pred * d + c  
            coarse = coarse * d + c
            index = pointnet2.furthest_point_sample(pred, npoints * args.up_ratio)
            index_1 = pointnet2.furthest_point_sample(coarse, npoints * args.up_ratio)
            pred = pred.squeeze(0)[index.squeeze(0).long()]
            coarse = coarse.squeeze(0)[index_1.squeeze(0).long()]
            np.savetxt(
                os.path.join(out_folder_pred, name_list[i]), pred.cpu().numpy(), fmt="%.6f"
            )
            np.savetxt(
                os.path.join(out_folder_coarse, name_list[i]), coarse.cpu().numpy(), fmt="%.6f"
            )
            if gt_val_list is not None:
                gt = gt_val_list[i]
                gt = gt.unsqueeze(0).float().to(device)
            else:
                continue
            if pred.shape[0] == gt.shape[1]:
                pred, _, _ = pc_util.normalize_point_cloud(pred.cpu().numpy())
                pred = torch.from_numpy(pred).unsqueeze(0).contiguous().to(device)
                # 1 n 3
                cd_loss += val_loss.get_cd_loss(pred, gt).item()
                hd_loss += val_loss.get_hd_loss(pred, gt).item()
    print(
        "cd loss : {:.4f}, hd loss : {:.4f}".format(
            (cd_loss / num_sample * 1000), (hd_loss / num_sample * 1000)
        )
    )
    print(
        "Avenge Inference time is :{:.6f}ms".format(np.mean(infer_times[1:]) * 1000.0)
    )
    print("It spend :{:.6f}s".format(time() - start))
    torch.cuda.empty_cache()
    return cd_loss / num_sample, hd_loss / num_sample

def _init_fn(work_id):
    np.random.seed(args.seed + work_id)

def set_seed():
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
set_seed()

num_gpu = len(args.gpu)
device = torch.device("cuda")
Loss_fn = Loss()
checkpoint_path = os.path.join(args.log_dir, args.checkpoint_path)
checkpoint = None
lr = args.base_lr
logger = Logger(args)


(
    npoints,
    input_val_list,
    gt_val_list,
    centroid_val_list,
    distance_val_list,
    name_list,
) = pc_util.get_val_data(args)

if args.phase == "test" or args.restore:
    print("=> loading checkpoint '{}' ... ".format(checkpoint_path), end="")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint["epoch"] + 1
    print('start epoch', start_epoch)
    logger.best_result = checkpoint["best_result"]
else:
    start_epoch = 0
model = Model(args).to(device)

model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model_named_params, lr=lr, betas=(args.beta1, args.beta2))   

if checkpoint:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> checkpoint state loaded.")
else:
    model.apply(xavier_init)
model = torch.nn.DataParallel(model)

if args.phase == "train":
    args.code_dir = os.path.join(args.log_dir, "code")
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.code_dir).mkdir(parents=True, exist_ok=True)
    # ===> save scripts
    if not os.path.exists(os.path.join(args.code_dir, 'network')):
        shutil.copytree('network', os.path.join(args.code_dir, 'network'))
    train_dataset = PUGAN_Dataset(args)
    train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=_init_fn,
        )
    n_set = len(train_data_loader)
    print('this is train_data_loader', n_set)   
    for epoch in range(start_epoch, args.training_epoch):
        model.train()
        is_best = False
        lr = adjust_learning_rate(args, epoch, optimizer)
        gamma = adjust_gamma(args.fidelity_feq, epoch)
        for idx, (input, gt, radius) in enumerate(train_data_loader):
            start = time()
            optimizer.zero_grad()
            input = input.reshape(-1, args.num_point, 3)   
            gt = gt.reshape(-1, args.num_point * args.up_ratio, 3)  
            radius = radius.reshape(-1) 
            input = input.permute(0, 2, 1).contiguous().float().to(device)
            gt = gt.permute(0, 2, 1).contiguous().float().to(device)
            radius = radius.float().to(device)
            gpu_start = time()
            sparse, refine = model(input)  
            gpu_time = time() - gpu_start
            sparse = sparse.permute(0, 2, 1).contiguous()
            refine = refine.permute(0, 2, 1).contiguous()
            gt = gt.permute(0, 2, 1).contiguous()
            if args.use_repulse:
                repulsion_loss = args.repulsion_w * Loss_fn.get_repulsion_loss(refine)
            else:
                repulsion_loss = torch.tensor(0.0)
            if args.use_uniform:
                uniform_loss = args.uniform_w * Loss_fn.get_uniform_loss(
                    refine, radius=radius
                )
            else:
                uniform_loss = torch.tensor(0.0)
            if args.use_l2:
                L2_loss = Loss_fn.get_l2_regular_loss(model, args.regular_w)
            else:
                L2_loss = torch.tensor(0.0)
            if args.use_hd:
                hd_loss = args.hd_w * Loss_fn.get_hd_loss(refine, gt, radius)
            else:
                hd_loss = torch.tensor(0.0)
            if args.use_emd:
                sparse_loss = args.fidelity_w * Loss_fn.get_emd_loss(sparse, gt, radius)
                refine_loss = args.fidelity_w * Loss_fn.get_emd_loss(refine, gt, radius)
            else:
                sparse_loss = args.fidelity_w * Loss_fn.get_cd_loss(sparse, gt, radius)
                refine_loss = args.fidelity_w * Loss_fn.get_cd_loss(refine, gt, radius)

            loss = (
                gamma * refine_loss
                + gamma * hd_loss
                + sparse_loss
                + repulsion_loss
                + uniform_loss
                + L2_loss
            )
            loss.backward()
            optimizer.step()
            step = epoch * n_set + idx
            logger.save_info(
                lr,
                gamma,
                repulsion_loss.item(),
                uniform_loss.item(),
                sparse_loss.item(),
                refine_loss.item(),
                L2_loss.item(),
                hd_loss.item(),
                loss.item(),
                step,
            )
            total_time = time() - start
            logger.print_info(
                gpu_time,
                total_time,
                sparse_loss.item(),
                refine_loss.item(),
                L2_loss.item(),
                hd_loss.item(),
                loss.item(),
                epoch,
                step,
            )  
        if epoch > args.start_eval_epoch:
            print('this is epoch > 0 test', len(input_val_list), len(gt_val_list), len(centroid_val_list),
                  len(distance_val_list))
            model.eval()
            cd, hd = test(
                model,
                npoints,
                input_val_list,
                gt_val_list,
                centroid_val_list,
                distance_val_list,
                name_list,
            )

            logger.save_val_data(epoch, cd, hd)
            if logger.best_result > cd:
                logger.best_result = cd
                is_best = True
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "best_result": logger.best_result,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.is_save_all,
            epoch,
            args.log_dir,
        )
    os.system('shutdown')

else:
    model.eval()
    test(
        model,
        npoints,
        input_val_list,
        gt_val_list,
        centroid_val_list,
        distance_val_list,
        name_list,
    )
    # r = 4
    if args.n_upsample == 2:
        args.test_dir = os.path.join(args.log_dir, args.out_dir)
        args.out_dir = "{}_up_2".format(args.out_dir)
        if args.gt_dir != "":
            args.gt_dir = "data/PUGAN/test/gt_32768/"
        (
            npoints,
            input_val_list,
            gt_val_list,
            centroid_val_list,
            distance_val_list,
            name_list,
        ) = pc_util.get_val_data(args)
        test(
            model,
            npoints,
            input_val_list,
            gt_val_list,
            centroid_val_list,
            distance_val_list,
            name_list,
        )
    # r = 16

