# Option 1 with no dual slow/fast causal pathway

# Segmentation only
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version sum_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version convnet_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version conv3d_fusion &
#python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version global_atten_fusion &

# Depth only
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version sum_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version convnet_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version conv3d_fusion &
#python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version global_atten_fusion &

# Multi-task only
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version sum_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version conv3d_fusion &
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version causal_fusion &

# Option 1 with no dual slow/fast causal pathway

# Segmentation only
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version sum_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version convnet_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version conv3d_fusion --causal True &
#python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version global_atten_fusion &

# Depth only
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version sum_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version convnet_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_depth --version conv3d_fusion --causal True &
#python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version global_atten_fusion &

# Multi-task only
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version sum_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version conv3d_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_both --version causal_fusion --causal True &






# SLOW/FAST ENOCDERS PREVIOUS MODEL (IGNORE)

# run the temporal segmentations baselines python
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version sum_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version convnet_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version conv3d_fusion &
##python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version global_atten_fusion &
#
## Run for depth if time permits
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_depth --version sum_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_depth --version convnet_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_depth --version conv3d_fusion &
##python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_depth --version global_atten_fusion &
#
## run with unsupervision + segmentation
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version sum_fusion --unsup True &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version convnet_fusion --unsup True &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version conv3d_fusion --unsup True &
##python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_seg --version global_atten_fusion --unsup True &
#
## Run for multi-task learning setup, | segmentation w/ versions | depth no versions
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version sum_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version conv3d_fusion &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version causal_fusion &
#
## Run with unsupervised loss on the encoders
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version sum_fusion --unsup True &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version conv3d_fusion --unsup True &
#python train_temporal_baseline.py --config configs/medtronic_cluster/temporal_cityscape_config_both --version causal_fusion --unsup True &




