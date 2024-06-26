#!/bin/bash
usage() {
  echo "Usage: ${0} [-g|--gpu] [-c|--case] [-s|--sparse_weight]  [-lr|--learning_rate]  [-lr_geo|--learning_rate_geo]"  1>&2
  exit 1
}
while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -c|--case)
      CASE=${2}
      shift 2
      ;;
    -g|--gpu)
      GPU=${2}
      shift 2
      ;;
    -s|--sparse_weight)
      SPARSE_WEIGHT=${2}
      shift 2
      ;;
    -lr|--learning_rate)
      LR=${2}
      shift 2
      ;;
    -lr_geo|--learning_rate_geo)
      LR_GEO=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

getenv=True
source /home/yxiu/miniconda3/bin/activate neuraludf

CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode validate_image \
--conf ./confs/udf_garment_blending_mask_ft.conf \
--case ${CASE} --is_continue --resolution 512

CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode extract_udf_mesh \
--conf ./confs/udf_garment_blending_mask_ft.conf \
--case ${CASE} --is_continue --resolution 512

CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode validate_mesh \
--conf ./confs/udf_garment_blending_mask_ft.conf \
--case ${CASE} --is_continue --resolution 512

# CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode validate_image \
# --conf ./confs/udf_garment_blending_mask.conf \
# --case ${CASE} --is_continue --resolution 512

# CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode extract_udf_mesh \
# --conf ./confs/udf_garment_blending_mask.conf \
# --case ${CASE} --is_continue --resolution 512

# CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --mode validate_mesh \
# --conf ./confs/udf_garment_blending_mask.conf \
# --case ${CASE} --is_continue --resolution 512