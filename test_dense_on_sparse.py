import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import utils.loss
import utils.data
import utils.improc
import utils.misc
import utils.saveload
from tensorboardX import SummaryWriter
import datetime
import time

torch.set_float32_matmul_precision('medium')

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params += param
    # print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

def get_dataset(dname, args):
    if dname=='bad':
        dataset_names = ['bad']
        from datasets import badjadataset
        dataset = badjadataset.BadjaDataset(
            data_root=os.path.join(args.dataset_root, 'badja2'),
            crop_size=args.image_size,
            only_first=False, 
        ) 
    elif dname=='cro':
        dataset_names = ['cro']
        from datasets import crohddataset
        dataset = crohddataset.CrohdDataset(
            data_root=os.path.join(args.dataset_root, 'crohd'),
            crop_size=args.image_size,
            seq_len=None,
            only_first=True,
        )
    elif dname=='dav':
        dataset_names = ['dav']
        from datasets import davisdataset
        dataset = davisdataset.DavisDataset(
            data_root=os.path.join(args.dataset_root, 'tapvid_davis'),
            crop_size=args.image_size,
            only_first=False, 
        )
    elif dname=='dri':
        dataset_names = ['dri']
        from datasets import drivetrackdataset
        dataset = drivetrackdataset.DrivetrackDataset(
            data_root=os.path.join(args.dataset_root, 'drivetrack'),
            crop_size=args.image_size,
            seq_len=None,
            traj_per_sample=768,
            only_first=True, 
        )
    elif dname=='ego':
        dataset_names = ['ego']
        from datasets import egopointsdataset
        dataset = egopointsdataset.EgoPointsDataset(
            data_root=os.path.join(args.dataset_root, 'ego_points'),
            crop_size=args.image_size,
            only_first=True, 
        )
    elif dname=='hor':
        dataset_names = ['hor']
        from datasets import horsedataset
        dataset = horsedataset.HorseDataset(
            data_root=os.path.join(args.dataset_root, 'horse10'),
            crop_size=args.image_size,
            seq_len=None,
            only_first=True, 
        )
    elif dname=='kin':
        dataset_names = ['kin']
        from datasets import kineticsdataset
        dataset = kineticsdataset.KineticsDataset(
            data_root=os.path.join(args.dataset_root, 'tapvid_kinetics'),
            crop_size=args.image_size, 
            only_first=True, 
        )
    elif dname=='rgb':
        dataset_names = ['rgb']
        from datasets import rgbstackingdataset
        dataset = rgbstackingdataset.RGBStackingDataset(
            data_root=os.path.join(args.dataset_root, 'tapvid_rgb_stacking'),
            crop_size=args.image_size,
            only_first=False, 
        )
    elif dname=='rob':
        dataset_names = ['rob']
        from datasets import robotapdataset
        dataset = robotapdataset.RobotapDataset(
            data_root=os.path.join(args.dataset_root, 'robotap'),
            crop_size=args.image_size,
            only_first=True, 
        )
    return dataset, dataset_names

def create_pools(args, n_pool=10000, min_size=1): 
    pools = {}
    n_pool = max(n_pool, 10)
    thrs = [1,2,4,8,16]
    for thr in thrs:
        pools['d_%d' % thr] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
        pools['jac_%d' % thr] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['d_avg'] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['aj'] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    pools['oa'] = utils.misc.SimplePool(n_pool, version='np', min_size=min_size)
    return pools

def forward_batch(batch, model, args, sw):
    rgbs = batch.video
    trajs_g = batch.trajs
    vis_g = batch.visibs
    valids = batch.valids
    dname = batch.dname
    # print('rgbs', rgbs.shape, rgbs.dtype, rgbs.device)
    # print('trajs_g', trajs_g.shape, trajs_g.device)
    # print('vis_g', vis_g.shape, vis_g.device)
    
    B, T, C, H, W = rgbs.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = rgbs.device
    assert(B==1)

    trajs_g = trajs_g.cuda()
    vis_g = vis_g.cuda()
    valids = valids.cuda()
    __, first_positive_inds = torch.max(vis_g, dim=1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

    trajs_e = torch.zeros([B, T, N, 2], device='cuda:0')
    visconfs_e = torch.zeros([B, T, N, 2], device='cuda:0')
    query_points_all = []
    with torch.no_grad():
        for first_positive_ind in torch.unique(first_positive_inds):
            chunk_pt_idxs = torch.nonzero(first_positive_inds[0]==first_positive_ind, as_tuple=False)[:, 0]  # K
            chunk_pts = trajs_g[:, first_positive_ind[None].repeat(chunk_pt_idxs.shape[0]), chunk_pt_idxs]  # B, K, 2
            query_points_all.append(torch.cat([first_positive_inds[:, chunk_pt_idxs, None], chunk_pts], dim=2))
            
            traj_maps_e = grid_xy.repeat(1,T,1,1,1) # B,T,2,H,W
            visconf_maps_e = torch.zeros_like(traj_maps_e)
            if first_positive_ind < T-1:
                if T > 128: # forward_sliding is a little safer memory-wise
                    forward_flow_e, forward_visconf_e, forward_flow_preds, forward_visconf_preds = \
                        model.forward_sliding(rgbs[:, first_positive_ind:], iters=args.inference_iters, sw=sw, is_training=False)
                else: 
                    forward_flow_e, forward_visconf_e, forward_flow_preds, forward_visconf_preds = \
                        model(rgbs[:, first_positive_ind:], iters=args.inference_iters, sw=sw, is_training=False)

                del forward_flow_preds
                del forward_visconf_preds
                forward_traj_maps_e = forward_flow_e.cuda() + grid_xy # B,Tf,2,H,W, when T = 2, flow has no T dim, but we broadcast
                traj_maps_e[:,first_positive_ind:] = forward_traj_maps_e
                visconf_maps_e[:,first_positive_ind:] = forward_visconf_e
                if not sw.save_this:
                    del forward_flow_e
                    del forward_visconf_e
                    del forward_traj_maps_e
                
            xyt = trajs_g[:,first_positive_ind].round().long()[0, chunk_pt_idxs] # K,2
            trajs_e_chunk = traj_maps_e[:, :, :, xyt[:,1], xyt[:,0]] # B,T,2,K
            trajs_e_chunk = trajs_e_chunk.permute(0,1,3,2) # B,T,K,2
            trajs_e.scatter_add_(2, chunk_pt_idxs[None, None, :, None].repeat(1, trajs_e_chunk.shape[1], 1, 2), trajs_e_chunk)

            visconfs_e_chunk = visconf_maps_e[:, :, :, xyt[:,1], xyt[:,0]] # B,T,2,K
            visconfs_e_chunk = visconfs_e_chunk.permute(0,1,3,2) # B,T,K,2
            visconfs_e.scatter_add_(2, chunk_pt_idxs[None, None, :, None].repeat(1, visconfs_e_chunk.shape[1], 1, 2), visconfs_e_chunk)

    visconfs_e[..., 0] *= visconfs_e[..., 1]
    assert (torch.all(visconfs_e >= 0) and torch.all(visconfs_e <= 1))
    vis_thr = 0.6
    query_points_all = torch.cat(query_points_all, dim=1)[..., [0, 2, 1]]
    gt_occluded = (vis_g < .5).bool().transpose(1, 2)
    gt_tracks = trajs_g.transpose(1, 2)
    pred_occluded = (visconfs_e[..., 0] < vis_thr).bool().transpose(1, 2)
    pred_tracks = trajs_e.transpose(1, 2)
    
    metrics = utils.misc.compute_tapvid_metrics(
        query_points=query_points_all.cpu().numpy(),
        gt_occluded=gt_occluded.cpu().numpy(),
        gt_tracks=gt_tracks.cpu().numpy(),
        pred_occluded=pred_occluded.cpu().numpy(),
        pred_tracks=pred_tracks.cpu().numpy(),
        query_mode='first',
        crop_size=args.image_size
    )
    for thr in [1, 2, 4, 8, 16]:
        metrics['d_%d' % thr] = metrics['pts_within_' + str(thr)]
        metrics['jac_%d' % thr] = metrics['jaccard_' + str(thr)]
    metrics['d_avg'] = metrics['average_pts_within_thresh']
    metrics['aj'] = metrics['average_jaccard']
    metrics['oa'] = metrics['occlusion_accuracy']
    
    return metrics


def run(dname, model, args):
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)
    seed = 42
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    B_ = args.batch_size * torch.cuda.device_count()
    assert(B_==1)
    model_name = "%dx%d" % (int(args.image_size[0]), int(args.image_size[1]))
    model_name += "i%d" % (args.inference_iters)
    model_name += "_%s" % args.init_dir
    model_name += "_%s" % dname
    if args.only_first:
        model_name += "_first"
    model_name += "_%s" % args.exp
    model_date = datetime.datetime.now().strftime('%M%S')
    model_name = model_name + '_' + model_date

    save_dir = '%s/%s' % (args.ckpt_dir, model_name)

    dataset, dataset_names = get_dataset(dname, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,
        collate_fn=utils.data.collate_fn_train,
    )
    iterloader = iter(dataloader)
    print('len(dataloader)', len(dataloader))

    log_dir = './logs_test_dense_on_sparse'
    overpools_t = create_pools(args)
    writer_t = SummaryWriter(log_dir + '/' + args.model_type + '-' + model_name + '/t', max_queue=10, flush_secs=60)
    
    global_step = 0
    if args.init_dir:
        load_dir = '%s/%s' % (args.ckpt_dir, args.init_dir)
        _ = utils.saveload.load(
            None,
            load_dir,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()

    max_steps = min(args.max_steps, len(dataloader))

    while global_step < max_steps:
        torch.cuda.empty_cache()
        iter_start_time = time.time()
        try:
            batch = next(iterloader)
        except StopIteration:
            iterloader = iter(dataloader)
            batch = next(iterloader)

        batch, gotit = batch
        if not all(gotit):
            continue
        
        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=args.log_freq,
            fps=8,
            scalar_freq=1,
            just_gif=True)
        if args.log_freq == 9999:
            sw_t.save_this = False

        rtime = time.time()-iter_start_time
        
        if batch.trajs.shape[2] == 0:
            global_step += 1
            continue

        metrics = forward_batch(batch, model, args, sw_t)

        # update stats
        for key in list(overpools_t.keys()):
            if key in metrics:
                overpools_t[key].update([metrics[key]])
        # plot stats
        for key in list(overpools_t.keys()):
            sw_t.summ_scalar('_/%s' % (key), overpools_t[key].mean())

        global_step += 1

        itime = time.time()-iter_start_time

        info_str = '%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_steps, rtime, itime)
        info_str += '; dname %s; d_avg %.1f aj %.1f oa %.1f' % (
            dname, overpools_t['d_avg'].mean()*100.0, overpools_t['aj'].mean()*100.0, overpools_t['oa'].mean()*100.0
        )
        if sw_t.save_this:
            print('model_name', model_name)

        if not args.print_less:
            print(info_str, flush=True)
            

    if args.print_less:
        print(info_str, flush=True)

    writer_t.close()

    del iterloader
    del dataloader
    del dataset

    return overpools_t['d_avg'].mean()*100.0, overpools_t['aj'].mean()*100.0, overpools_t['oa'].mean()*100.0

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    init_dir = ''

    exp = ''

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=exp)
    parser.add_argument("--dname", type=str, nargs='+', default=None, help="Dataset names, written as a single string or list of strings")
    parser.add_argument("--init_dir", type=str, default=init_dir)
    parser.add_argument("--ckpt_dir", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--log_freq", type=int, default=9999)
    parser.add_argument("--dataset_root", type=str, default='/orion/group')
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--image_size", nargs="+", default=[384, 512]) # resizing arg
    parser.add_argument("--backwards", default=False)
    parser.add_argument("--mixed_precision", action='store_true', default=False)
    parser.add_argument("--only_first", action='store_true', default=False)
    parser.add_argument("--no_split", action='store_true', default=False)
    parser.add_argument("--print_less", action='store_true', default=False)
    parser.add_argument("--use_basicencoder", action='store_true', default=False)
    parser.add_argument("--conf", action='store_true', default=False)
    parser.add_argument("--model_type", choices=['ours', 'raft', 'searaft', 'accflow', 'delta'], default='ours')
    args = parser.parse_args()
    # allow dname to be a comma-separated string (e.g., "rgb,bad,dav")
    if args.dname is not None and len(args.dname) == 1 and ',' in args.dname[0]:
        args.dname = args.dname[0].split(',')
    if args.dname is None:
        args.dname = ['bad', 'cro', 'dav', 'dri', 'hor', 'kin', 'rgb', 'rob']
    dataset_names = args.dname
    args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
    full_start_time = time.time()

    from nets.alltracker import Net; model = Net(16)
    url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=True)
    print('loaded weights from', url)
    
    das, ajs, oas = [], [], []
    for dname in dataset_names:
        if dname==dataset_names[0]:
            count_parameters(model)
        da, aj, oa = run(dname, model, args)
        das.append(da)
        ajs.append(aj)
        oas.append(oa)
    for (data, name) in zip([dataset_names, das, ajs, oas], ['dn', 'da', 'aj', 'oa']):
        st = name + ': '
        for dat in data:
            if isinstance(dat, str):
                st += '%s,' % dat
            else:
                st += '%.1f,' % dat
        print(st)
    full_time = time.time()-full_start_time
    print('full_time %.1f' % full_time)        
