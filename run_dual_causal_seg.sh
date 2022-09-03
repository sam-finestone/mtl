python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version sum_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version convnet_fusion --causal True &
python train_temporal_baseline.py --config configs/felix_temporal_config_seg --version conv3d_fusion --causal True