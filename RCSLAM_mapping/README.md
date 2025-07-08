<<<<<<< HEAD
# ZZULI-FIRE-LAB
RC-SLAM
=======

RC-SLAM: Normal-Guided Implicit Neural SLAM

RC-SLAM is a SLAM system that integrates normal-guided pre-reconstruction and IMU-based optimization into an implicit neural mapping framework. It improves 3D reconstruction accuracy, consistency, and robustness. The system introduces geometric priors via normal constraints, fuses IMU data to correct LiDAR drift, and adopts hierarchical feature encoding for scalable mapping.

Key Contributions:

Normal-Guided Pre-Reconstruction: Uses normal estimation and plane fitting to generate square constraints as geometric priors, reducing distortion and improving consistency.

IMU-Based Optimization: Fuses IMU data to refine both pose and normal estimation, enhancing robustness under motion and compensating for LiDAR sampling bias.

Hierarchical Feature Encoding: Employs an octree-based structure with positional encoding to efficiently represent features across scales.

Comprehensive Evaluation: Outperforms baseline methods in accuracy, completeness, and memory efficiency across multiple datasets.

Quick Start:

Install Dependencies:
Make sure Python >= 3.8 and CUDA >= 11.6 are available. Install packages:
pip install -r requirements.txt

Configure Dataset:
Edit config file:

pc_path: folder with point clouds (.bin / .ply / .pcd)

pose_path: file with 4x4 poses

imu_path: IMU measurements (optional)

Run RC-SLAM:
For batch mapping:
python rc_slam_batch.py config/rcslam_batch.yaml

For incremental mapping:
python rc_slam_incre.py config/rcslam_incre.yaml

Output files:

mesh/*.ply for reconstructed mesh

model/ for trained decoder

map/ for optional voxel-based map (if enabled)

Evaluation:

To evaluate reconstruction results:
python eval/evaluator.py

This outputs Chamfer distance, completeness, accuracy, and F-score using ground truth point clouds.


Contact:

For questions, contact:
zhouzhihui@wti.ac.cn
>>>>>>> 5764f74 (first commit)
