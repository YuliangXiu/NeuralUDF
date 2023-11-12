import numpy as np
import os
import cv2 as cv
import torch
import glob
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio

psnr = PeakSignalNoiseRatio()

# method = "NeuralUDF"
# method = "NeUDF"
method = "NeAT"
# method = "NeuS"

root_dir = f"/home/yxiu/Code/{method}"
# subjects  = ["30", "92", "117", "133", "164", "320", "448", "522", "591"]
subjects = [0,2,4,6,8,10]
psnr_lst = []

out_path = f"metric/{method}-psnr.npy"

if not os.path.exists(out_path):
    for subject in subjects:
        
        if method == "NeuralUDF":
            res = 1024
            img_dir = os.path.join(root_dir, 
                                f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_specular_mask/novel_view")
            pbar = tqdm(glob.glob(os.path.join(img_dir, 'pred*.png')))

        elif method == "NeUDF":
            res=256
            img_dir = os.path.join(root_dir, 
                               f"exp/{subject}/wmask_specular_open/validations_fine")
            pbar = tqdm(glob.glob(os.path.join(img_dir, '00400000_0*.png')))
            
        elif method == "NeAT":
            res=1024
            img_dir = os.path.join(root_dir, 
                               f"exp/{subject}/wmask_specular/validations_fine")
            pbar = tqdm(glob.glob(os.path.join(img_dir, '00400000_0*[!_mask].png')))
        elif method == "NeuS":
            res=1024
            img_dir = os.path.join(root_dir, 
                               f"exp/{subject}/specular_wmask/validations_fine")
            pbar = tqdm(glob.glob(os.path.join(img_dir, '00400000_0*[!_mask].png')))
        
        pred_mat = torch.zeros(200, 3, res, res)
        gt_mat = torch.zeros(200, 3, res, res)
        
        for idx, pred_file in enumerate(pbar):
            pbar.set_description(f"{method}-{subject}")
            if method == "NeuralUDF":
                gt_file = pred_file.replace('pred', 'gt')
                pred_img = cv.imread(os.path.join(img_dir, pred_file)) / 255.0
                gt_img = cv.imread(os.path.join(img_dir, gt_file)) / 255.0
            elif method in ["NeUDF","NeAT","NeuS"]:
                full_img = cv.imread(os.path.join(img_dir, pred_file)) / 255.0
                pred_img, gt_img = np.split(full_img, 2)
                
            pred_mat[idx] = torch.tensor(pred_img.transpose(2,0,1))
            gt_mat[idx] = torch.tensor(gt_img.transpose(2,0,1))

        psnr_value = psnr(pred_mat, gt_mat).item()
        psnr_lst.append(psnr_value)

    np.save(out_path, psnr_lst)
else:
    psnr_lst = np.load(out_path).tolist()
    
psnr_avg = np.mean(psnr_lst)
psnr_lst.append(psnr_avg)
psnr_str = ""
for data in psnr_lst:
    psnr_str+= f"{data:.2f} & " 

print(f"{method}: {psnr_str}")
    