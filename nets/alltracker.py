import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc
import numpy as np

from nets.blocks import CNBlockConfig, ConvNeXt, conv1x1, RelUpdateBlock, InputPadder, CorrBlock, BasicEncoder

class Net(nn.Module):
    def __init__(
            self,
            seqlen,
            use_attn=True,
            use_mixer=False,
            use_conv=False,
            use_convb=False,
            use_basicencoder=False,
            use_sinmotion=False,
            use_relmotion=False,
            use_sinrelmotion=False,
            use_feats8=False,
            no_time=False,
            no_space=False,
            no_split=False,
            no_ctx=False,
            full_split=False,
            corr_levels=5,
            corr_radius=4,
            num_blocks=3,
            dim=128,
            hdim=128,
            init_weights=True,
    ):
        super(Net, self).__init__()

        self.dim = dim
        self.hdim = hdim

        self.no_time = no_time
        self.no_space = no_space
        self.seqlen = seqlen
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_channel = self.corr_levels * (self.corr_radius * 2 + 1) ** 2
        self.num_blocks = num_blocks

        self.use_feats8 = use_feats8
        self.use_basicencoder = use_basicencoder
        self.use_sinmotion = use_sinmotion
        self.use_relmotion = use_relmotion
        self.use_sinrelmotion = use_sinrelmotion
        self.no_split = no_split
        self.no_ctx = no_ctx
        self.full_split = full_split

        if use_basicencoder:
            if self.full_split:
                self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
                self.cnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
            else:
                if self.no_split:
                    self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
                else:
                    self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim*2, stride=8)
        else:
            block_setting = [
                CNBlockConfig(96, 192, 3, True), # 4x
                CNBlockConfig(192, 384, 3, False), # 8x
                CNBlockConfig(384, None, 9, False), # 8x
            ]
            self.cnn = ConvNeXt(block_setting, stochastic_depth_prob=0.0, init_weights=init_weights)
            if self.no_split:
                self.dot_conv = conv1x1(384, dim)
            else:
                self.dot_conv = conv1x1(384, dim*2)
            
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim, 2, kernel_size=3, padding=1)
        )
        self.visconf_head = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim, 2, kernel_size=3, padding=1)
        )

        if self.use_sinrelmotion:
            self.pdim = 84 # 32*2
        elif self.use_relmotion:
            self.pdim = 4
        elif self.use_sinmotion:
            self.pdim = 42
        else:
            self.pdim = 2
            
        self.update_block = RelUpdateBlock(self.corr_channel, self.num_blocks, cdim=dim, hdim=hdim, pdim=self.pdim,
                                           use_attn=use_attn, use_mixer=use_mixer, use_conv=use_conv, use_convb=use_convb,
                                           use_layer_scale=True, no_time=no_time, no_space=no_space,
                                           no_ctx=no_ctx)

        time_line = torch.linspace(0, seqlen-1, seqlen).reshape(1, seqlen, 1)
        self.register_buffer("time_emb", utils.misc.get_1d_sincos_pos_embed_from_grid(self.dim, time_line[0])) # 1,S,C

        
    def fetch_time_embed(self, t, dtype, is_training=False):
        S = self.time_emb.shape[1]
        if t == S:
            return self.time_emb.to(dtype)
        elif t==1:
            if is_training:
                ind = np.random.choice(S)
                return self.time_emb[:,ind:ind+1].to(dtype)
            else:
                return self.time_emb[:,1:2].to(dtype)
        else:
            time_emb = self.time_emb.float()
            time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1) 
            return time_emb.to(dtype)
    
    def coords_grid(self, batch, ht, wd, device, dtype):
        coords = torch.meshgrid(torch.arange(ht, device=device, dtype=dtype), torch.arange(wd, device=device, dtype=dtype), indexing='ij')
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = self.coords_grid(N, H//8, W//8, device=img.device)
        coords2 = self.coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W).to(flow.dtype)

    def get_T_padded_images(self, images, T, S, is_training, stride=None, pad=True):
        B,T,C,H,W = images.shape
        indices = None
        if T > 2:
            step = S // 2 if stride is None else stride
            indices = []
            start = 0
            while start + S < T:
                indices.append(start)
                start += step
            indices.append(start)
            Tpad = indices[-1]+S-T
            if pad:
                if is_training:
                    assert Tpad == 0
                else:
                    images = images.reshape(B,1,T,C*H*W)
                    if Tpad > 0:
                        padding_tensor = images[:,:,-1:,:].expand(B,1,Tpad,C*H*W)
                        images = torch.cat([images, padding_tensor], dim=2)
                    images = images.reshape(B,T+Tpad,C,H,W)
                    T = T+Tpad
        else:
            assert T == 2
        return images, T, indices

    def get_fmaps(self, images_, B, T, sw, is_training):
        _, _, H_pad, W_pad = images_.shape # revised HW

        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        if self.no_split:
            C = self.dim

        fmaps_chunk_size = 64
        if (not is_training) and (T > fmaps_chunk_size):
            images = images_.reshape(B,T,3,H_pad,W_pad)
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                images_chunk = images[:, t : t + fmaps_chunk_size]
                images_chunk = images_chunk.cuda()
                if self.use_basicencoder:
                    if self.full_split:
                        fmaps_chunk1 = self.fnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                        fmaps_chunk2 = self.cnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                        fmaps_chunk = torch.cat([fmaps_chunk1, fmaps_chunk2], axis=1)
                    else:
                        fmaps_chunk = self.fnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                else:
                    fmaps_chunk = self.cnn(images_chunk.reshape(-1, 3, H_pad, W_pad))
                    if t==0 and sw is not None and sw.save_this:
                        sw.summ_feat('1_model/fmap_raw', fmaps_chunk[0:1])
                    fmaps_chunk = self.dot_conv(fmaps_chunk) # B*T,C,H8,W8
                T_chunk = images_chunk.shape[1]
                fmaps.append(fmaps_chunk.reshape(B, -1, C, H8, W8))
            fmaps_ = torch.cat(fmaps, dim=1).reshape(-1, C, H8, W8)
        else:
            if not is_training:
                # sometimes we need to move things to cuda here
                images_ = images_.cuda()
            if self.use_basicencoder:
                if self.full_split:
                    fmaps1_ = self.fnet(images_)
                    fmaps2_ = self.cnet(images_)
                    fmaps_ = torch.cat([fmaps1_, fmaps2_], axis=1)
                else:
                    fmaps_ = self.fnet(images_)
            else:
                fmaps_ = self.cnn(images_)
                if sw is not None and sw.save_this:
                    sw.summ_feat('1_model/fmap_raw', fmaps_[0:1])
                fmaps_ = self.dot_conv(fmaps_) # B*T,C,H8,W8
        return fmaps_
    
    def forward(self, images, iters=4, sw=None, is_training=False, stride=None):
        B,T,C,H,W = images.shape
        S = self.seqlen
        device = images.device
        dtype = images.dtype

        # images are in [0,255]
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = images / 255.0
        images = (images - mean)/std

        T_bak = T
        if stride is not None:
            pad = False
        else:
            pad = True
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride=stride, pad=pad)

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape # revised HW
        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        C2 = C//2
        if self.no_split:
            C = self.dim
            C2 = C

        fmaps = self.get_fmaps(images_, B, T, sw, is_training).reshape(B,T,C,H8,W8)
        device = fmaps.device

        fmap_anchor = fmaps[:,0]

        if T<=2 or is_training:
            # note: collecting preds can get expensive on a long video
            all_flow_preds = []
            all_visconf_preds = []
        else:
            all_flow_preds = None
            all_visconf_preds = None

        if T > 2: # multiframe tracking
            
            # we will store our final outputs in these tensors
            full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            # 1/8 resolution 
            full_flows8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            full_visconfs8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

            if self.use_feats8:
                full_feats8 = torch.zeros((B,T,C2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visits = np.zeros((T))

            for ii, ind in enumerate(indices):
                ara = np.arange(ind,ind+S)
                if ii < len(indices)-1:
                    next_ind = indices[ii+1]
                    next_ara = np.arange(next_ind,next_ind+S)
                
                # print("torch.cuda.memory_allocated: %.1fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), 'ara', ara)
                fmaps2 = fmaps[:,ara]
                flows8 = full_flows8[:,ara].reshape(B*(S),2,H_pad//8,W_pad//8).detach()
                visconfs8 = full_visconfs8[:,ara].reshape(B*(S),2,H_pad//8,W_pad//8).detach()

                if self.use_feats8:
                    if ind==0:
                        feats8 = None
                    else:
                        feats8 = full_feats8[:,ara].reshape(B*(S),C2,H_pad//8,W_pad//8).detach()
                else:
                    feats8 = None

                flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                    fmap_anchor, fmaps2, visconfs8, iters=iters, flowfeat=feats8, flows8=flows8,
                    is_training=is_training)

                unpad_flow_predictions = []
                unpad_visconf_predictions = []
                for i in range(len(flow_predictions)):
                    flow_predictions[i] = padder.unpad(flow_predictions[i])
                    unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                    visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                    unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

                full_flows[:,ara] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)
                full_flows8[:,ara] = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
                full_visconfs[:,ara] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)
                full_visconfs8[:,ara] = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)
                if self.use_feats8:
                    full_feats8[:,ara] = feats8.reshape(B,S,C2,H_pad//8,W_pad//8)
                visits[ara] += 1

                if is_training:
                    all_flow_preds.append(unpad_flow_predictions)
                    all_visconf_preds.append(unpad_visconf_predictions)
                else:
                    del unpad_flow_predictions
                    del unpad_visconf_predictions

                # for the next iter, replace empty data with nearest available preds
                invalid_idx = np.where(visits==0)[0]
                valid_idx = np.where(visits>0)[0]
                for idx in invalid_idx:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                    # print('replacing %d with %d' % (idx, nearest))
                    full_flows8[:,idx] = full_flows8[:,nearest]
                    full_visconfs8[:,idx] = full_visconfs8[:,nearest]
                    if self.use_feats8:
                        full_feats8[:,idx] = full_feats8[:,nearest]
        else: # flow

            flows8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

            flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                fmap_anchor, fmaps[:,1:2], visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)
            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            full_flows = all_flow_preds[-1].reshape(B,2,H,W)
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W)
                
        if (not is_training) and (T > 2):
            full_flows = full_flows[:,:T_bak]
            full_visconfs = full_visconfs[:,:T_bak]
            
        return full_flows, full_visconfs, all_flow_preds, all_visconf_preds
    
    def forward_sliding(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None):
        B,T,C,H,W = images.shape
        S = self.seqlen if window_len is None else window_len
        device = images.device
        dtype = images.dtype
        stride = S // 2 if stride is None else stride

        # images are in [0,255]
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = images / 255.0
        images = (images - mean)/std

        T_bak = T
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride)
        assert stride <= S // 2

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape # revised HW
        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        C2 = C//2
        if self.no_split:
            C = self.dim
            C2 = C
            
        all_flow_preds = None
        all_visconf_preds = None
        
        if T<=2:
            # note: collecting preds can get expensive on a long video
            all_flow_preds = []
            all_visconf_preds = []
            
            fmaps = self.get_fmaps(images_, B, T, sw, is_training).reshape(B,T,C,H8,W8)
            device = fmaps.device
            
            flows8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                
            fmap_anchor = fmaps[:,0]
            
            flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                fmap_anchor, fmaps[:,1:2], visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)
            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            full_flows = all_flow_preds[-1].reshape(B,2,H,W).detach().cpu()
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W).detach().cpu()
            
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds

        assert T > 2 # multiframe tracking
        
        if is_training:
            all_flow_preds = []
            all_visconf_preds = []
            
        # we will store our final outputs in these cpu tensors
        full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')
        full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')
        
        images_ = images_.reshape(B,T,3,H_pad,W_pad)
        fmap_anchor = self.get_fmaps(images_[:,:1].reshape(-1,3,H_pad,W_pad), B, 1, sw, is_training).reshape(B,C,H8,W8)
        device = fmap_anchor.device
        full_visited = torch.zeros((T,), dtype=torch.bool, device=device)

        for ii, ind in enumerate(indices):
            ara = np.arange(ind,ind+S)
            if ii == 0:
                flows8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                visconfs8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                fmaps2 = self.get_fmaps(images_[:,ara].reshape(-1,3,H_pad,W_pad), B, S, sw, is_training).reshape(B,S,C,H8,W8)
            else:
                flows8 = torch.cat([flows8[:,stride:stride+S//2], flows8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                visconfs8 = torch.cat([visconfs8[:,stride:stride+S//2], visconfs8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                fmaps2 = torch.cat([fmaps2[:,stride:stride+S//2], 
                                    self.get_fmaps(images_[:,np.arange(ind+S//2,ind+S)].reshape(-1,3,H_pad,W_pad), B, S//2, sw, is_training).reshape(B,S//2,C,H8,W8)], dim=1)
            
            flows8 = flows8.reshape(B*S,2,H_pad//8,W_pad//8).detach()
            visconfs8 = visconfs8.reshape(B*S,2,H_pad//8,W_pad//8).detach()
            
            flow_predictions, visconf_predictions, flows8, visconfs8, _ = self.forward_window(
                fmap_anchor, fmaps2, visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)

            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

            current_visiting = torch.zeros((T,), dtype=torch.bool, device=device)
            current_visiting[ara] = True
            
            to_fill = current_visiting & (~full_visited)
            to_fill_sum = to_fill.sum().item()
            full_flows[:,to_fill] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)[:,-to_fill_sum:].detach().cpu()
            full_visconfs[:,to_fill] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)[:,-to_fill_sum:].detach().cpu()
            full_visited |= current_visiting

            if is_training:
                all_flow_preds.append(unpad_flow_predictions)
                all_visconf_preds.append(unpad_visconf_predictions)
            else:
                del unpad_flow_predictions
                del unpad_visconf_predictions
                
            flows8 = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
            visconfs8 = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)
                
        if not is_training:
            full_flows = full_flows[:,:T_bak]
            full_visconfs = full_visconfs[:,:T_bak]
            
        return full_flows, full_visconfs, all_flow_preds, all_visconf_preds
        
    def forward_window(self, fmap1_single, fmaps2, visconfs8, iters=None, flowfeat=None, flows8=None, sw=None, is_training=False):
        B,S,C,H8,W8 = fmaps2.shape
        device = fmaps2.device
        dtype = fmaps2.dtype

        flow_predictions = []
        visconf_predictions = []

        fmap1 = fmap1_single.unsqueeze(1).repeat(1,S,1,1,1) # B,S,C,H,W
        fmap1 = fmap1.reshape(B*(S),C,H8,W8).contiguous()

        fmap2 = fmaps2.reshape(B*(S),C,H8,W8).contiguous()

        visconfs8 = visconfs8.reshape(B*(S),2,H8,W8).contiguous()

        corr_fn = CorrBlock(fmap1, fmap2, self.corr_levels, self.corr_radius)

        coords1 = self.coords_grid(B*(S), H8, W8, device=fmap1.device, dtype=dtype)

        if self.no_split:
            flowfeat, ctxfeat = fmap1.clone(), fmap1.clone()
        else:
            if flowfeat is not None:
                _, ctxfeat = torch.split(fmap1, [self.dim, self.dim], dim=1)
            else:
                flowfeat, ctxfeat = torch.split(fmap1, [self.dim, self.dim], dim=1)
                
        # add pos emb to ctxfeat (and not flowfeat), since ctxfeat is untouched across iters
        time_emb = self.fetch_time_embed(S, ctxfeat.dtype, is_training).reshape(1,S,self.dim,1,1).repeat(B,1,1,1,1)
        ctxfeat = ctxfeat + time_emb.reshape(B*S,self.dim,1,1)

        if self.no_ctx:
            flowfeat = flowfeat + time_emb.reshape(B*S,self.dim,1,1)
            
        for itr in range(iters):
            _, _, H8, W8 = flows8.shape
            flows8 = flows8.detach()
            coords2 = (coords1 + flows8).detach() # B*S,2,H,W
            corr = corr_fn(coords2).to(dtype)

            if self.use_relmotion or self.use_sinrelmotion:
                coords_ = coords2.reshape(B,S,2,H8*W8).permute(0,1,3,2) # B,S,H8*W8,2
                rel_coords_forward = coords_[:, :-1] - coords_[:, 1:]
                rel_coords_backward = coords_[:, 1:] - coords_[:, :-1]
                rel_coords_forward = torch.nn.functional.pad(
                    rel_coords_forward, (0, 0, 0, 0, 0, 1) # pad the 3rd-last dim (S) by (0,1)
                )
                rel_coords_backward = torch.nn.functional.pad(
                    rel_coords_backward, (0, 0, 0, 0, 1, 0) # pad the 3rd-last dim (S) by (1,0)
                )
                rel_coords = torch.cat([rel_coords_forward, rel_coords_backward], dim=-1) # B,S,H8*W8,4

                if self.use_sinrelmotion:
                    rel_pos_emb_input = utils.misc.posenc(
                        rel_coords,
                        min_deg=0,
                        max_deg=10,
                    )  # B,S,H*W,pdim
                    motion = rel_pos_emb_input.reshape(B*S,H8,W8,self.pdim).permute(0,3,1,2).to(dtype) # B*S,pdim,H8,W8
                else:
                    motion = rel_coords.reshape(B*S,H8,W8,4).permute(0,3,1,2).to(dtype) # B*S,4,H8,W8
                
            else:
                if self.use_sinmotion:
                    pos_emb_input = utils.misc.posenc(
                        flows8.reshape(B,S,H8*W8,2),
                        min_deg=0,
                        max_deg=10,
                    )  # B,S,H*W,pdim
                    motion = pos_emb_input.reshape(B*S,H8,W8,self.pdim).permute(0,3,1,2).to(dtype) # B*S,pdim,H8,W8
                else:
                    motion = flows8
                    
            flowfeat = self.update_block(flowfeat, ctxfeat, visconfs8, corr, motion, S)
            flow_update = self.flow_head(flowfeat)
            visconf_update = self.visconf_head(flowfeat)
            weight_update = .25 * self.upsample_weight(flowfeat)
            flows8 = flows8 + flow_update
            visconfs8 = visconfs8 + visconf_update
            flow_up = self.upsample_data(flows8, weight_update)
            flow_predictions.append(flow_up)
            visconf_up = self.upsample_data(visconfs8, weight_update)
            visconf_predictions.append(visconf_up)
            
        return flow_predictions, visconf_predictions, flows8, visconfs8, flowfeat

    

