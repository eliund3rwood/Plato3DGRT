#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Several functions in this codebase were repurposed from nerf-pytorch 
# https://github.com/yenchenlin/nerf-pytorch
# nerf-pytorch is released under the MIT license.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as scisig
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from utils.load_tof import load_tof_data
from utils.nerf_helpers import *

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "3DGRUT"))
from threedgrt_tracer.tracer import Tracer, load_3dgrt_plugin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

SPEED_OF_LIGHT = 2.99792458e8
TIME_RES_M = 0.0384
TIME_RES_S = 0.0384 / SPEED_OF_LIGHT
FWHM_TO_SIGMA = 2.35482004503

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def batchify_rays(rays_flat, chunk=1024*32, debug_title="", per_ray_far=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], debug_title=debug_title, per_ray_far=per_ray_far, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(rays,
           chunk=1024*32,
           c2w=None,
           ndc=True,
           near=0.,
           far=6.,
           use_viewdirs=False,
           c2w_staticcam=None,
           debug_title="",
           per_ray_far=None,
           **kwargs):
    """Render rays
    Args:
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      intensity_map: [batch_size, 1]. Predicted intensity values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    # use provided ray batch
    rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, debug_title=debug_title, per_ray_far=per_ray_far, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['intensity_map', 'disp_map', 'acc_map', 'depth_map', 'trans']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_rays(ray_batch, tracer,
                particle_density, mog_sph,
                per_ray_far=None,
                **kwargs):

    rays_o = ray_batch[:,0:3]
    rays_d = ray_batch[:,3:6]

    if per_ray_far is not None:
        max_dist = per_ray_far
    else:
        max_dist = ray_batch[:,7]

    ray_radiance, ray_density, ray_hit_distance, _, _, _ = tracer.trace(
        0,
        None,
        rays_o,
        rays_d,
        particle_density,
        mog_sph,
        None,
        3,
        1e-4
    )

    if DEBUG:
        assert ray_hit_distance.requires_grad

    depth = ray_hit_distance
    trans = torch.exp(-ray_density)

    ret = {
        "intensity_map": torch.zeros_like(depth)[:,None],
        "disp_map": 1/(depth+1e-6),
        "acc_map": 1-trans,
        "depth_map": depth,
        "trans": trans,
        "depth0": depth,
        "trans0": trans,
        "pts": rays_o[:,None,:] + depth[:,None,None]*rays_d[:,None,:]
    }

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--use_all_rays", type=int, default=0,
                        help='whether or not to use all rays')
    parser.add_argument("--per_image_thresh", type=float, action='append', required=True)
    parser.add_argument("--debug", type=int, default=0,
                        help='whether or not to debug')
    parser.add_argument("--near", type=float, default=0.1,
                        help='near plane')
    parser.add_argument("--dist_weight", type=int, default=1000,
                        help='dist weight')
    parser.add_argument("--extract_first", type=int, default=0,
                        help='whether or not to extract 1b distance to compute projected illumination')
    parser.add_argument("--parallel", type=float, default=0.05,
                        help='parallel filter')
    parser.add_argument("--shadw", type=float, default=1.0,
                        help='shadow loss weight for shadow pixels')
    parser.add_argument("--nonshadw", type=float, default=1.0,
                        help='shadow loss weight for non shadow pixels')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--downsample", type=int, default=1,
                        help='downsample rays by factor of x')
    parser.add_argument("--downsample_temp", type=int, default=0,
                        help='downsample rays by factor of x')
    parser.add_argument("--save_lights", type=int, default=0,
                        help='save light paths')
    parser.add_argument("--vis_rays", type=int, default=0,
                        help='visualize ray paths via a video; 1 true, 0 false')


    # training options
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian noise on time of arrival")
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin'),
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging'),
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving'),
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving'),
    parser.add_argument("--simple_mode", action='store_true',
                        help='whether or not to only render 1B to 2B rays'),
    parser.add_argument("--sigmoid", action='store_true',
                        help='whether or not to apply sigmoid on shadows'),
    parser.add_argument("--bce", action='store_false',
                        help='whether or not to use BCE loss'),

    return parser


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def find_ray_intersections(rays, walls):
    pixels = []
    intersections = []
    distances = []
    for idx in range(rays.shape[0]):
        ray_o = rays[idx,0]
        ray_d = rays[idx,1]
        plane_normal = np.array(walls[idx,:3])
        plane_point = np.array(walls[idx,3:6])
        x = walls[idx, 6]
        min_y = walls[idx, 7]
        max_y = walls[idx, 8]
        min_z = walls[idx, 9]
        max_z = walls[idx, 10]

        intersection_point = LinePlaneCollision(plane_normal, plane_point, ray_d, ray_o)
        if intersection_point is not None and \
                intersection_point[1] >= min_y and intersection_point[1] <= max_y and \
                intersection_point[2] >= min_z and intersection_point[2] <= max_z:
                    pixels.append([idx])
                    intersections.append(intersection_point)
                    distances.append(np.linalg.norm(ray_o - intersection_point))
    return pixels, intersections, distances

def find_ray_intersections_from_tof(rays, tof):
    """ Uses first bounce return to compute where the intersection of light and scene occurs """
    pixels = []
    intersections = []
    distances = []
    for i in range(rays.shape[0]):
        rays_o, rays_d = rays[i][0], rays[i][1]
        tof_i = np.reshape(tof[i], [-1, tof[i].shape[-1]])
        intensities = np.max(tof_i, axis=1)
        binidxs = np.argmax(tof_i, axis=1)
        idx = binidxs[np.argmax(intensities)]
        distance = ((idx + 1) * TIME_RES_M) / 2.0
        intersection = rays_o + (distance * rays_d)
        pixels.append([i])
        intersections.append(intersection)
        distances.append(distance)
    return pixels, intersections, distances
                
EPSILON = 1e-5
def normalize_min_max(tensor, new_max=1.0, new_min=0.0):
     return (tensor - tensor.min())/(tensor.max() - tensor.min() + EPSILON)*(new_max - new_min) + new_min

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == "dtof":
        tof, poses, light_o, light_d, hwf, walls_cam, walls_light = load_tof_data(args.datadir, args.ignore)
        print('Loaded ToF data', tof.shape, light_o.shape, light_d.shape, hwf, args.datadir, walls_cam.shape, walls_light.shape)
        render_poses = torch.tensor([0.0]).float()
        i_train = np.arange(tof.shape[0])
        i_val = []
        i_test = []

        print("Train idxs: {}".format(i_train))

        near = args.near
        far = 6.0

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Gaussian initialization
    N_GAUSS = 50000
    scene_radius = 3.0

    mog_pos = torch.rand(N_GAUSS,3, device=device)*2*scene_radius - scene_radius
    mog_dns = torch.ones(N_GAUSS,1, device=device)*0.1
    mog_rot = torch.zeros(N_GAUSS,4, device=device)
    mog_rot[:,0] = 1.0
    mog_scl = torch.ones(N_GAUSS,3, device=device)*0.05
    mog_sph = torch.zeros(N_GAUSS,16, device=device)

    for p in [mog_pos, mog_dns, mog_rot, mog_scl, mog_sph]:
        p.requires_grad_()

    # Create tracer
    conf = {
        "num_particles": N_GAUSS,
        "device": "cuda"
    }

    load_3dgrt_plugin(conf)
    tracer = Tracer(conf)

    # Optimizer
    optimizer = torch.optim.Adam(
        [mog_pos, mog_dns, mog_rot, mog_scl, mog_sph],
        lr=args.lrate
    )

    # Renderer params
    render_kwargs_train = {
        "tracer": tracer,
        "mog_pos": mog_pos,
        "mog_dns": mog_dns,
        "mog_rot": mog_rot,
        "mog_scl": mog_scl,
        "mog_sph": mog_sph,
        "near": near,
        "far": far
    }

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    # For random ray batching
    print('get rays')
    poses[:,2,:] *= -1
    poses[:,0,:] *= -1
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]

    # we need this if using subset of illumination points in i_train
    light_o = np.stack([light_o[i] for i in i_train],0)
    light_d = np.stack([light_d[i] for i in i_train],0)
    walls_cam = np.stack([walls_cam[i] for i in i_train],0)
    walls_light = np.stack([walls_light[i] for i in i_train],0)
    tof = np.stack([tof[i] for i in i_train],0)

    rays = np.transpose(rays, [0,2,3,1,4]) # [N, H, W, ro+rd+lo+ld, 3]
    rays = np.stack([rays[i] for i in i_train], 0) # train cameras only
    rays = np.reshape(rays, [-1,2,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays = rays.astype(np.float32)

    # normalizing ray directions!
    rays_d = rays[:,1,:]
    norm = np.linalg.norm(rays[:,1,:], axis=1)
    rays[:,1] = rays_d / norm[:,None]

    walls_cam = np.tile(walls_cam, H*W).reshape((walls_cam.shape[0], H, W, walls_cam.shape[1])) 
    walls_cam = np.reshape(walls_cam, [-1, walls_cam.shape[-1]])

    i_batch = 0

    """ light intersections """
    light_rays = np.stack([light_o, light_d],1)
    light_idx, light_inters, light_dists = None, None, None
    if args.extract_first == 1:
        light_idx, light_inters, light_dists = find_ray_intersections_from_tof(light_rays, tof)
    else:
        # this should only be used for debugging - directly compute light intersection based on known geometry
        light_idx, light_inters, light_dists = find_ray_intersections(light_rays, walls_light)
    light_inters = np.array(light_inters)

    if args.save_lights == 1:
        np.save("{}/lights.npy".format(os.path.join(basedir, expname)), light_inters)
        exit()

    print("The following projected lights have been found:", light_idx)
    assert len(light_idx) == len(i_train)
    light_inters = np.tile(light_inters, H*W).reshape((light_inters.shape[0], H, W, light_inters.shape[1]))
    light_inters = np.reshape(light_inters, [-1, light_inters.shape[-1]])

    light_dists = np.array(light_dists)
    light_dists = np.tile(light_dists[:,None], H*W).reshape((light_dists.shape[0], H, W))
    light_dists = np.reshape(light_dists, [-1, 1])

    light_dirs = np.tile(light_d, H*W).reshape((light_d.shape[0], H, W, light_d.shape[1]))
    light_dirs = np.reshape(light_dirs, [-1, light_dirs.shape[-1]])

    """ Filter out rays near illumination spot + separate shadows w/ matched filter """

    # Filter out rays close to illumination
    parallel = np.linalg.norm(np.cross(rays[:,1,:], light_dirs), axis=1)
    mask = parallel < args.parallel
    mask = np.array(mask, dtype=np.uint8)

    b1 = parallel < 0.005
    b1 = np.array(b1, dtype=np.uint8)
    b1 = np.reshape(b1, [light_d.shape[0], H, W])

    tof = np.reshape(tof, [-1, tof.shape[-1]])
    tof = tof.astype(np.float32)

    noise = np.zeros([tof.shape[0],])
    if args.noise != 0:
        noise_m = 1e-12 * args.noise * SPEED_OF_LIGHT
        noise = np.random.normal(0.0, noise_m / FWHM_TO_SIGMA, tof.shape[0])
        print("adding noise of {} m (FWHM)! Resulting min {}, max {} m.".format(noise_m, np.min(noise), np.max(noise)))
    noise = noise[::args.downsample]
    
    tof = np.reshape(tof, [light_d.shape[0], H, W, -1])
    mask = np.reshape(mask, [light_d.shape[0], H, W])
    tof_stack = []

    """ Matched filter code is slow --> we can opt not to run it for debugging, but note tof will not be usable for training """
    if args.debug == 0:
        tof_1b_norm = 0

        shadow_savedir = os.path.join(basedir, expname, "shadows")
        os.makedirs(shadow_savedir, exist_ok=True)
        for i, tofi in enumerate(tof):
            print("Preprocessing tof image {} of {}.".format(i+1, tof.shape[0]))
            tof_1b = tofi[b1[i] == 1.0][0]
            tof_1b_norm = tof_1b / np.sum(tof_1b) 

            tofi[mask[i]==1.0] = 0.0 # Filter 1st bounce returns based on angle
            corr = np.zeros([H,W])
            for pixeli in range(H):
                for pixelj in range(W):
                    pix = tofi[pixeli][pixelj]
                    if np.sum(pix) != 0:
                        pix = pix / np.sum(pix)

                    cc = scisig.correlate(pix, tof_1b_norm)
                    corr[pixeli,pixelj] = np.max(cc)

            thresh = args.per_image_thresh[i]
            corr[corr < thresh] = 0
            corr[corr >= thresh] = 1
            arrivals = np.multiply(tofi, np.expand_dims(corr,2))
            tof_stack.append(arrivals)

            target_shadow = np.sum(arrivals, axis=2)
            target_shadow[target_shadow > 0.0] = 1.0
            target_shadow = np.stack([target_shadow, target_shadow, target_shadow],axis=2)
            target_shadow = np.float32(target_shadow)
            target_shadow = cv2.cvtColor(target_shadow, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("{}/shadow_{}.png".format(shadow_savedir, str(i).zfill(3)), target_shadow*255)

        tof = np.stack(tof_stack, 0)
 
    tof = np.reshape(tof, [-1, tof.shape[-1]])
    tof = tof.astype(np.float32)
    rays = rays[::args.downsample]
    print("Rays: {}, Downsample: {}".format(rays.shape[0], args.downsample))
    light_inters = light_inters[::args.downsample]
    print("Light inters: {}, Downsample: {}".format(light_inters.shape[0], args.downsample))
    light_dists = light_dists[::args.downsample]
    print("Light dists: {}, Downsample: {}".format(light_dists.shape[0], args.downsample))
    light_dirs = light_dirs[::args.downsample]
    print("Light dirs: {}, Downsample: {}".format(light_dirs.shape[0], args.downsample))
    tof = tof[::args.downsample]
    print("ToF: {}, Downsample: {}".format(tof.shape[0], args.downsample))


    # Ablation on downsampling temporal resolution
    if args.downsample_temp:
        tof_down = np.zeros([tof.shape[0], int(tof.shape[1] / args.downsample_temp)])
        print("Integrating transient from shape {} to {} and temporal res {} to {}".format(tof.shape, tof_down.shape, TIME_RES_M, TIME_RES_M*args.downsample_temp))
        for i in range(0, tof_down.shape[1], 1):
            tof_down[:,i] = np.sum(tof[:,(i*args.downsample_temp):((i+1)*args.downsample_temp)], axis=1)
        tof = tof_down
        globals()["TIME_RES_M"] = TIME_RES_M * args.downsample_temp
        globals()["TIME_RES_S"] = TIME_RES_S * args.downsample_temp
        print(tof.shape, TIME_RES_M, TIME_RES_S)

    """ Permute all data used in training """
    print('shuffle rays and transients in unison')
    p = np.random.permutation(len(rays))
    tof = tof[p]
    rays = rays[p]
    light_inters = light_inters[p]
    light_dists = light_dists[p]
    light_dirs = light_dirs[p]
    print('done')
        
    """ Filter out shadow rays """
    tof_sum = np.sum(tof, axis=1)
    indices = np.where(tof_sum == 0)[0]
    ftof = np.delete(tof, indices, axis=0)
    flight_inters = np.delete(light_inters, indices, axis=0)
    flight_dists = np.delete(light_dists, indices, axis=0)
    frays = np.delete(rays, indices, axis=0)
    fnoise = np.delete(noise, indices, axis=0)

    tof = torch.Tensor(tof)
    light_inters = torch.Tensor(light_inters)
    light_dists = torch.Tensor(light_dists)
    rays = torch.Tensor(rays)
    noise = torch.Tensor(noise)

    ftof = torch.Tensor(ftof)
    flight_inters = torch.Tensor(flight_inters)
    flight_dists = torch.Tensor(flight_dists)
    frays = torch.Tensor(frays)
    fnoise = torch.Tensor(fnoise)

    n_iters_pretrain = 25000
    DIST_WEIGHT = args.dist_weight

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    criterion = torch.nn.MSELoss()

    start = 0
    global_step = 0

    for i in trange(start, N_iters):
        time0 = time.time()

        # Random over all images
        # For first N iterations, only train on non-shadow rays with distance supervision
        if i < n_iters_pretrain:
            dataset_size = flight_inters.shape[0]
            target_tof = ftof[i_batch:i_batch+N_rand].to(device)
            batch_lights = flight_inters[i_batch:i_batch+N_rand].to(device)
            batch_light_dists = flight_dists[i_batch:i_batch+N_rand].squeeze().to(device)
            batch_rays = frays[i_batch:i_batch+N_rand,:,:]
            batch_rays = torch.transpose(batch_rays, 0, 1).to(device)
            batch_noise = fnoise[i_batch:i_batch+N_rand].to(device)
        else:
            dataset_size = light_inters.shape[0]
            target_tof = tof[i_batch:i_batch+N_rand].to(device)
            batch_lights = light_inters[i_batch:i_batch+N_rand].to(device)
            batch_light_dists = light_dists[i_batch:i_batch+N_rand].squeeze().to(device)
            batch_rays = rays[i_batch:i_batch+N_rand,:,:]
            batch_rays = torch.transpose(batch_rays, 0, 1).to(device)
            batch_noise = noise[i_batch:i_batch+N_rand].to(device)

        target_dist_idx = torch.argmax(target_tof, dim=1)
        target_dist_idx[target_dist_idx > 0] += 1
        target_dist = ((target_dist_idx * TIME_RES_M) + batch_noise.squeeze()) / 15.0 # distance supervision
        target_shadow = torch.sum(target_tof, dim=1) # transmittance probability supervision
        target_shadow[target_shadow > 0.0] = 1.0

        secondary_idxs = torch.arange(0,1024, device=device)
        shadow_idxs = torch.where(target_shadow == 0.0)[0]
        nonshadow_idxs = torch.where(target_shadow == 1.0)[0]

        if i < n_iters_pretrain:
            target_pred = target_dist
        else:
            target_pred = torch.stack([target_dist,target_shadow],dim=1)

        i_batch += N_rand
        if i_batch >= dataset_size:
            if i < n_iters_pretrain:
                rand_idx = torch.randperm(flight_inters.shape[0])
                ftof = ftof[rand_idx]
                flight_inters = flight_inters[rand_idx]
                flight_dists = flight_dists[rand_idx]
                frays = frays[rand_idx]
                fnoise = fnoise[rand_idx]
            else:
                rand_idx = torch.randperm(light_inters.shape[0])
                tof = tof[rand_idx]
                light_inters = light_inters[rand_idx]
                light_dists = light_dists[rand_idx]
                rays = rays[rand_idx]
                noise = noise[rand_idx]
            i_batch = 0

        particle_density = torch.cat([mog_pos, mog_dns, mog_rot, mog_scl], dim=1)
        render_kwargs_train["particle_density"] = particle_density

        #####  Core optimization loop  #####
        """ camera to first bounce """
        intensity, disp, acc, depth, trans, extras = render(batch_rays, chunk=args.chunk,
                                              verbose=i < 10, retraw=True, debug_title="0_bounce",
                                              **render_kwargs_train)

        batch_vray_o = batch_rays[0] + torch.mul(batch_rays[1], depth[:,None])
        batch_vray_term = batch_lights
        batch_vray_d = batch_vray_term - batch_vray_o
        norm = torch.norm(batch_vray_d, dim=1)
        batch_vray_d = batch_vray_d / norm[:,None]
        batch_vrays = torch.stack([batch_vray_o, batch_vray_d], 0)

        total_distance_1 = (depth + norm + batch_light_dists) / 15.0

        if i < n_iters_pretrain:
            loss_1 = criterion(total_distance_1, target_pred)
        else:
            """ first bounce to second bounce """
            batch_vrays = batch_vrays[:,secondary_idxs,:]
            norm = norm[secondary_idxs]
            _, _, _, b_depth, b_trans, b_extras = render(batch_vrays, chunk=args.chunk,
                                                             verbose=i < 10, retraw=True,
                                                             debug_title="1_bounce",
                                                             per_ray_far=norm,
                                                             **render_kwargs_train)

            """ Visualize all rays (creates a video) """
            if args.vis_rays:
                import matplotlib.pyplot as plt
                from matplotlib import animation
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = Axes3D(fig)

                def init():
                    for idx in range(extras['pts'].shape[0]):
                        start = batch_rays[0][idx]
                        stop = extras['pts'][idx][-1]
                        line = torch.stack([start, stop], dim=0)
                        line = line.detach().cpu().numpy()
                        ax.plot3D(line[:,0], line[:,2], line[:,1], 'gray')

                        start = batch_vrays[0][idx]
                        direction = batch_vrays[1][idx]
                        stop = b_extras['pts'][idx][-1]
                        line = torch.stack([start, stop], dim=0)
                        line = line.detach().cpu().numpy()
                        ax.plot3D(line[:,0], line[:,2], line[:,1], 'red')
                        if idx > 100:
                            break
                    return fig,

                def animate(i):
                    ax.view_init(elev=10., azim=i)
                    return fig,

                # Animate
                anim = animation.FuncAnimation(fig, animate, init_func=init,
                                               frames=360, interval=20, blit=True)
                # Save
                anim.save('ray_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
                exit()

            if "depth0" in b_extras:
                # 4 losses: dist 1, dist 2, shadow 1, shadow 2
                # filter out shadow rays from dist loss
                shad_loss_a_shad = criterion(target_pred[shadow_idxs,1], b_trans[shadow_idxs])
                shad_loss_b_shad = criterion(target_pred[shadow_idxs,1], b_extras["trans0"][shadow_idxs])
                shad_loss_a_nonshad = criterion(target_pred[nonshadow_idxs,1], b_trans[nonshadow_idxs])
                shad_loss_b_nonshad = criterion(target_pred[nonshadow_idxs,1], b_extras["trans0"][nonshadow_idxs])

                target_tof_sum = torch.sum(target_tof, dim=1)
                mask = target_tof_sum > 0
                dist_loss = criterion(total_distance_1[mask], target_pred[mask,0])
                
                shad_loss = (args.shadw * (shad_loss_a_shad + shad_loss_b_shad)) + (args.nonshadw * (shad_loss_a_nonshad + shad_loss_b_nonshad))
                loss_1 = (DIST_WEIGHT * dist_loss) + shad_loss
                loss_1_dist = dist_loss
                loss_1_shad = shad_loss
            else:
                # 2 losses: dist, shad
                shad_loss_shad = criterion(target_pred[shadow_idxs,1], b_trans[shadow_idxs])
                shad_loss_nonshad = criterion(target_pred[nonshadow_idxs,1], b_trans[nonshadow_idxs])

                target_tof_sum = torch.sum(target_tof, dim=1)
                mask = target_tof_sum > 0
                dist_loss = criterion(total_distance_1[mask], target_pred[mask,0])

                shad_loss = (args.shadw * shad_loss_shad) + (args.nonshadw * shad_loss_nonshad)
                loss_1 = (DIST_WEIGHT * dist_loss) + shad_loss
                loss_1_dist = dist_loss
                loss_1_shad = shad_loss

        if 'depth0' in extras:
            depth0 = extras['depth0']
            batch_vray_o = batch_rays[0] + torch.mul(batch_rays[1], depth0[:,None])
            batch_vray_term = batch_lights
            batch_vray_d = batch_vray_term - batch_vray_o
            norm = torch.norm(batch_vray_d, dim=1)
            batch_vray_d = batch_vray_d / norm[:,None]
            batch_vrays = torch.stack([batch_vray_o, batch_vray_d], 0)

            total_distance_2 = (depth0 + norm + batch_light_dists) / 15.0

            if i < n_iters_pretrain:
                loss_2 = criterion(total_distance_2, target_pred)
            else:
                # compute virtual rays from first to second bounce
                batch_vrays = batch_vrays[:,secondary_idxs,:]
                norm = norm[secondary_idxs]
                _, _, _, b_depth_1, b_trans_1, b_extras_1 = render(batch_vrays, chunk=args.chunk,
                                                                               verbose=i < 10, retraw=True,
                                                                               per_ray_far=norm,
                                                                               **render_kwargs_train)
                if "depth0" in b_extras_1:
                    # 4 losses: dist 1, dist 2, shadow 1, shadow 2
                    # filter out shadow rays from dist loss
                    shad_loss_a_shad = criterion(target_pred[shadow_idxs,1], b_trans_1[shadow_idxs])
                    shad_loss_b_shad = criterion(target_pred[shadow_idxs,1], b_extras_1["trans0"][shadow_idxs])
                    shad_loss_a_nonshad = criterion(target_pred[nonshadow_idxs,1], b_trans_1[nonshadow_idxs])
                    shad_loss_b_nonshad = criterion(target_pred[nonshadow_idxs,1], b_extras_1["trans0"][nonshadow_idxs])

                    target_tof_sum = torch.sum(target_tof, dim=1)
                    mask = target_tof_sum > 0
                    dist_loss = criterion(total_distance_2[mask], target_pred[mask,0])
                    
                    shad_loss = (args.shadw * (shad_loss_a_shad + shad_loss_b_shad)) + (args.nonshadw * (shad_loss_a_nonshad + shad_loss_b_nonshad))
                    loss_2 = (DIST_WEIGHT * dist_loss) + shad_loss
                    loss_2_dist = dist_loss
                    loss_2_shad = shad_loss
                else:
                    # 2 losses: dist, shad
                    shad_loss_shad = criterion(target_pred[shadow_idxs,1], b_trans_1[shadow_idxs])
                    shad_loss_nonshad = criterion(target_pred[nonshadow_idxs,1], b_trans_1[nonshadow_idxs])

                    target_tof_sum = torch.sum(target_tof, dim=1)
                    mask = target_tof_sum > 0
                    dist_loss = criterion(total_distance_2[mask], target_pred[mask,0])

                    shad_loss = (args.shadw * shad_loss_shad) + (args.nonshadw * shad_loss_nonshad)
                    loss_2 = (DIST_WEIGHT * dist_loss) + shad_loss
                    loss_2_dist = dist_loss
                    loss_2_shad = shad_loss

        """ train! """
        optimizer.zero_grad()
        loss = 100 * (loss_1 + loss_2) # weight loss by 100 to prevent it from getting too small
        loss.backward()
        optimizer.step()

        weight_sum = 0
        grad_norm = 0

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'mog_pos': mog_pos.detach(),
                'mog_dns': mog_dns.detach(),
                'mog_rot': mog_rot.detach(),
                'mog_scl': mog_scl.detach(),
                'mog_sph': mog_sph.detach(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}, Weight Sum: {weight_sum}, Grad Norm: {grad_norm}")

        global_step += 1


if __name__=='__main__':

    train()
