import os
from shutil import copyfile
from tqdm import tqdm
import glob

# mode = "specular"
mode = "raw"

if mode == "raw":
    subject_list = ["30", "92", "117", "133", "164", "320", "448", "522", "591"]
else:
    subject_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# for method in ["NeAT", "NeuS", "NeUDF", "NeuralUDF"]:
# for method in ["NeuS", "NeUDF", "NeuralUDF"]:
for method in ["NeuralUDF"]:

    root_dir = f"/home/yxiu/Code/{method}"

    for subject in subject_list:
        # for subject in ["30"]:
        out_dir = f"/is/cluster/fast/yxiu/NeuralUDF_data/shared_results2/{subject}/{method}"
        os.makedirs(os.path.join(out_dir, "imgs"), exist_ok=True)

        if method == "NeuralUDF":

            if mode == "raw":
                img_dir = os.path.join(
                    root_dir,
                    f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_mask/novel_view"
                )
                mesh_path = os.path.join(
                    root_dir,
                    f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_ft_mask/udf_meshes/udf_res512_step50000.ply"
                )
            else:
                img_dir = os.path.join(
                    root_dir,
                    f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_specular_mask/novel_view"
                )
                mesh_path = os.path.join(
                    root_dir,
                    f"exp/udf/garment_sphere/{subject}/udf_garment_woblending_mixsample_specular_mask/udf_meshes/udf_res512_step400000.ply"
                )

            pbar = tqdm(glob.glob(os.path.join(img_dir, '*.png')))

        elif method == "NeUDF":
            if mode == "raw":
                img_dir = os.path.join(root_dir, f"exp/{subject}/wmask_open/novel_view")
                mesh_path = os.path.join(
                    root_dir, f"exp/{subject}/wmask_open/meshes/mu00400000.ply"
                )
            else:
                img_dir = os.path.join(root_dir, f"exp/{subject}/wmask_specular_open/novel_view")
                mesh_path = os.path.join(
                    root_dir, f"exp/{subject}/wmask_specular_open/meshes/mu00400000.ply"
                )
            pbar = tqdm(glob.glob(os.path.join(img_dir, '*.png')))

        elif method == "NeAT":
            if mode == "raw":
                img_dir = os.path.join(root_dir, f"exp/{subject}/wmask/novel_view")
                mesh_path = os.path.join(root_dir, f"exp/{subject}/wmask/meshes/00400000.ply")
            else:
                img_dir = os.path.join(root_dir, f"exp/{subject}/wmask_specular/novel_view")
                mesh_path = os.path.join(
                    root_dir, f"exp/{subject}/wmask_specular/meshes/00400000.ply"
                )
            pbar = tqdm(glob.glob(os.path.join(img_dir, '*.png')))
        elif method == "NeuS":
            if mode == "raw":
                img_dir = os.path.join(root_dir, f"exp/{subject}/garment_wmask/novel_view")
                mesh_path = os.path.join(
                    root_dir, f"exp/{subject}/garment_wmask/meshes/00400000.ply"
                )
            else:
                img_dir = os.path.join(root_dir, f"exp/{subject}/specular_wmask/novel_view")
                mesh_path = os.path.join(
                    root_dir, f"exp/{subject}/specular_wmask/meshes/00400000.ply"
                )
            pbar = tqdm(glob.glob(os.path.join(img_dir, '*.png')))

        copyfile(mesh_path, os.path.join(out_dir, "mesh.ply"))
        for pred_file in pbar:
            copyfile(pred_file, os.path.join(out_dir, "imgs", os.path.basename(pred_file)))
