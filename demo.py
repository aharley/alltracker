import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time

def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    framerate = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    print('framerate', framerate)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames, framerate

def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2) # N,T,2
    visibs = visibs.permute(1,0) # N,T
    N = trajs.shape[0]
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

    rgbs = rgbs * bkg_opacity # darken, to see the point tracks better
    
    opacity = 1.0
    if rate==1:
        radius = 1
        opacity = 0.9
    elif rate==2:
        radius = 1
    elif rate== 4:
        radius = 2
    elif rate== 8:
        radius = 4
    else:
        radius = 6
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5  # [N,2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N,3]
        N = xy.shape[0]
        cx = xy[:, 0].long()  # [N]
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x  # [N,D,D]
        y_grid = cy[:, None, None] + disp_y  # [N,D,D]
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]  # [K]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]  # [K]
        colors_valid = colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[
            :, valid
        ]  # [3, K]
        idx_flat = (y_valid * W + x_valid).long()  # [K]

        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)  # [3, H*W]
        weighted_colors = colors_valid * icon_weights  # [3, K]
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # [3, H, W]
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
    rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
    if bkg_opacity==0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return rgbs

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def forward_video(rgbs, framerate, model, args):
    
    B,T,C,H,W = rgbs.shape
    assert C == 3
    device = rgbs.device
    assert(B==1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

    torch.cuda.empty_cache()
    print('starting forward...')
    f_start_time = time.time()

    flows_e, visconf_maps_e, _, _ = \
        model.forward_sliding(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
    traj_maps_e = flows_e.cuda() + grid_xy # B,Tf,2,H,W
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = \
            model.forward_sliding(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
        backward_traj_maps_e = backward_flows_e.cuda() + grid_xy # B,Tb,2,H,W, reversed
        backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
        visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
    ftime = time.time()-f_start_time
    print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
    utils.basic.print_stats('traj_maps_e', traj_maps_e)
    utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

    # subsample to make the vis more readable
    rate = args.rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2

    xy0 = trajs_e[0,0].cpu().numpy()
    colors = utils.improc.get_2d_colors(xy0, H, W)

    fn = args.mp4_path.split('/')[-1].split('.')[0]
    rgb_out_f = './pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
    print('rgb_out_f', rgb_out_f)
    temp_dir = 'temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
    utils.basic.mkdir(temp_dir)
    vis = []

    frames = draw_pts_gpu(rgbs[0].to('cuda:0'), trajs_e[0], visconfs_e[0,:,:,1] > args.conf_thr,
                          colors, rate=rate, bkg_opacity=args.bkg_opacity)
    print('frames', frames.shape)

    if args.vstack:
        frames_top = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_top, frames], axis=1)
    elif args.hstack:
        frames_left = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
        frames = np.concatenate([frames_left, frames], axis=2)
    
    print('writing frames to disk')
    f_start_time = time.time()
    for ti in range(T):
        temp_out_f = '%s/%03d.jpg' % (temp_dir, ti)
        im = PIL.Image.fromarray(frames[ti])
        im.save(temp_out_f)#, "PNG", subsampling=0, quality=80)
    ftime = time.time()-f_start_time
    print('finished writing; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
        
    print('writing mp4')
    os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate %d -pattern_type glob -i "./%s/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (framerate, temp_dir, rgb_out_f))

    # # flow vis
    # rgb_out_f = './flow_vis.mp4'
    # temp_dir = 'temp_flow_vis'
    # utils.basic.mkdir(temp_dir)
    # vis = []
    # for ti in range(T):
    #     flow_vis = utils.improc.flow2color(flows_e[0:1,ti])
    #     vis.append(flow_vis)
    # for ti in range(T):
    #     temp_out_f = '%s/%03d.png' % (temp_dir, ti)
    #     im = PIL.Image.fromarray(vis[ti][0].permute(1,2,0).cpu().numpy())
    #     im.save(temp_out_f, "PNG", subsampling=0, quality=100)
    # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 1 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))
    
    return None

def run(model, args):
    log_dir = './logs_demo'
    
    global_step = 0

    if args.ckpt_init:
        _ = utils.saveload.load(
            None,
            args.ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print('loaded weights from', args.ckpt_init)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('loaded weights from', url)

    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()

    rgbs, framerate = read_mp4(args.mp4_path)
    print('rgbs[0]', rgbs[0].shape)
    H,W = rgbs[0].shape[:2]
    
    # shorten & shrink the video, in case the gpu is small
    if args.max_frames:
        rgbs = rgbs[:args.max_frames]
    HH = 1024
    scale = min(HH/H, HH/W)
    H, W = int(H*scale), int(W*scale)
    H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
    rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    print('rgbs[0]', rgbs[0].shape)

    # move to gpu
    rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float() # 1,T,C,H,W
    print('rgbs', rgbs.shape)
    
    with torch.no_grad():
        metrics = forward_video(rgbs, framerate, model, args)
    
    return None

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default='') # the ckpt we want (else default)
    parser.add_argument("--mp4_path", type=str, default='./demo_video/monkey.mp4') # input video 
    parser.add_argument("--query_frame", type=int, default=0) # which frame to track from
    parser.add_argument("--max_frames", type=int, default=400) # trim the video to this length
    parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
    parser.add_argument("--window_len", type=int, default=16) # model hyperparam
    parser.add_argument("--rate", type=int, default=2) # vis hyp
    parser.add_argument("--conf_thr", type=float, default=0.1) # vis hyp
    parser.add_argument("--bkg_opacity", type=float, default=0.5) # vis hyp
    parser.add_argument("--vstack", action='store_true', default=False) # whether to stack the input and output in the mp4
    parser.add_argument("--hstack", action='store_true', default=False) # whether to stack the input and output in the mp4
    args = parser.parse_args()

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
    
