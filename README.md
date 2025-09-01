# AllTracker: Efficient Dense Point Tracking at High Resolution

**[[Paper](https://arxiv.org/abs/2506.07310)] [[Project Page](https://alltracker.github.io/)] [[Gradio Demo](https://huggingface.co/spaces/aharley/alltracker)]**

<img src='https://alltracker.github.io/images/monkey.jpg'>

**AllTracker is a point tracking model which is faster and more accurate than other similar models, while also producing dense output at high resolution.**

AllTracker estimates long-range point tracks by way of estimating the flow field between a query frame and every other frame of a video. Unlike existing point tracking methods, our approach delivers high-resolution and dense (all-pixel) correspondence fields, which can be visualized as flow maps. Unlike existing optical flow methods, our approach corresponds one frame to hundreds of subsequent frames, rather than just the next frame.

We are actively adding to this repo, but please ping or open an issue if you notice something missing or broken. The demo (at least) should work for everyone!


## Env setup

Install miniconda:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init
```

Set up a fresh conda environment for AllTracker:

```
conda create -n alltracker python=3.12.8
conda activate alltracker
pip install -r requirements.txt
```

## Running the demo

Download the sample video:
```
cd demo_video
sh download_video.sh
cd ..
```

Run the demo:
```
python demo.py --mp4_path ./demo_video/monkey.mp4
```
The demo script will automatically download the model weights from [huggingface](https://huggingface.co/aharley/alltracker/tree/main) if needed.

For a fancier visualization, giving a side-by-side view of the input and output, try this:
```
python demo.py --mp4_path ./demo_video/monkey.mp4 --query_frame 32 --conf_thr 0.01 --bkg_opacity 0.0 --rate 2 --hstack --query_frame 16
```




## Training

AllTracker is trained in two stages: Stage 1 is kubric alone; Stage 2 is a mix of datasets. This 2-stage regime enables fair comparisons with models that train only on Kubric. 

### Data prep

Start by downloding Kubric. 

- 24-frame data: [kubric_au.tar.gz](https://huggingface.co/datasets/aharley/alltracker_data/resolve/main/kubric_au.tar.gz?download=true)

- 64-frame data: [part1](https://huggingface.co/datasets/aharley/alltracker_data/resolve/main/ce64_kub_aa?download=true), [part2](https://huggingface.co/datasets/aharley/alltracker_data/resolve/main/ce64_kub_ab?download=true), [part3](https://huggingface.co/datasets/aharley/alltracker_data/resolve/main/ce64_kub_ac?download=true)

Merge the parts by concatenating:
```
cat ce64_kub_aa ce64_kub_ab ce64_kub_ac > ce64_kub.tar.gz
```

The 24-frame Kubric data is a torch export of the official `kubric-public/tfds/movi_f/512x512` data.

With Kubric, you can skip the other datasets and start training Stage 1.

Download the rest of the point tracking datasets from [here](https://huggingface.co/datasets/aharley/alltracker_data/tree/main). There you will find 24-frame datasets, `ce24*.tar.gz`, and 64-frame datasets, `ce64*.tar.gz`. Some of the datasets are large, and they are split into parts, so you need to create the full files by concatenating. 

On disk, the point tracking datasets should look like this:
```
data/
├── ce24/
│   ├── drivingpt/
│   ├── fltpt/
│   ├── monkapt/
│   ├── springpt/
├── ce64/
│   ├── drivingpt/
│   ├── kublong/
│   ├── monkapt/
│   ├── podlong/
│   ├── springpt/
├── dynamicreplica/
├── kubric_au/
```

Download the optical flow datasets from the official websites: [FlyingChairs, FlyingThings3D, Monkaa, Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets) [AutoFlow](https://autoflow-google.github.io/), [SPRING](https://spring-benchmark.org/), [VIPER](https://playing-for-benchmarks.org/download/), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [TARTANAIR](https://theairlab.org/tartanair-dataset/). 


### Stage 1

Stage 1 is to train the model for 200k steps on Kubric. 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python train_stage1.py  --mixed_precision --lr 5e-4 --max_steps 200000 --data_dir /data --exp "stage1abc" 
```

This should produce a tensorboard log in `./logs_train/`, and checkpoints in `./checkpoints/`, in folder names similar to "64Ai4i3_5e-4m_stage1abc_1318". (The 4-digit string at the end is a timecode indicating when the run began, to help make the filepaths unique.)

### Stage 2

Stage 2 is to train the model for 400k steps on a mix of point tracking datasets and optical flow datasets. This stage initializes from the output of Stage 1.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python train_stage2.py  --mixed_precision --init_dir '64Ai4i3_5e-4m_stage1abc_1318' --lr 1e-5 --max_steps 400000 --exp 'stage2abc'
```

## Evaluation

Test the model on point tracking datasets with a command like:

```
python test_dense_on_sparse.py --dname 'dav'
```

At the end, you should see:
```
da: 76.3,
aj: 63.3,
oa: 90.0,
```
which represent `d_avg` (accuracy), Average Jaccard, and Occlusion Accuracy. We find that small numerical issues (even across GPUs) may cause +- 0.1 fluctuation on these metrics. 

Test it at higher resolution with the `image_size` arg, like: `--image_size 448 768`, which should produce `da: 78.8, aj: 65.9, oa: 90.2` or `--image_size 768 1024`, which should produce `da: 80.6, aj: 67.2, oa: 89.7`.

Note that AJ and OA are not reliable metrics in all datasets, because not all datasets follow the same rules about visibility annotation.

Dataloaders for all test datasets are in the repo, and can be run in sequence with a command like:
```
python test_dense_on_sparse.py --dname 'bad,cro,dav,dri,ego,hor,kin,rgb,rob'
```
but if you have multiple GPUs, we recomend running the tests in parallel.


## Citation

If you use this code for your research, please cite:

```
Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, Leonidas J. Guibas. AllTracker: Efficient Dense Point Tracking at High Resolution. ICCV 2025.
```

Bibtex:
```
@inproceedings{harley2025alltracker,
author    = {Adam W. Harley and Yang You and Xinglong Sun and Yang Zheng and Nikhil Raghuraman and Yunqi Gu and Sheldon Liang and Wen-Hsuan Chu and Achal Dave and Pavel Tokmakov and Suya You and Rares Ambrus and Katerina Fragkiadaki and Leonidas J. Guibas},
title     = {All{T}racker: {E}fficient Dense Point Tracking at High Resolution}
booktitle = {ICCV},
year      = {2025}
}
```
