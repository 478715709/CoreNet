import os
import tqdm
import json
from tools.visualize_nusc import NuScenes
use_gt = False
out_dir = './result_vis/'
result_json = "/mnt/share_disk/lyh/DBFusion-main/work_dirs/json/pred_instances_3d/results_nusc"
dataroot='/mnt/share_disk/lyh/DBFusion-main/data/nuscenes/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[1000:1100]):
    if use_gt:
        nusc.render_sample(token, out_path = "./result_vis/"+token+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = "./result_vis/"+token+"_pred.png", verbose=False)
    # import pdb
    # pdb.set_trace()