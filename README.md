# human_fitting
Fitting SMPL-X to RenderPeople mesh model.

## Example

[mesh_data]
- rp_aaron_posed_014_OBJ
  - rp_aaron_posed_014_100k.mtl
  - rp_aaron_posed_014_100k_textured.obj
  - rp_aaron_posed_014_dif.jpg

[data_root]
- rp_{subj_name} (subject_dir)
  - depth
  - images
  - keypts
  - skin
  - pcd.ply
  - pcd_body.ply
  - pcd_normal.npy
  - pose3d.npz
  - skin_label.npy

#### 0. Setting
```bash
$ docker pull jbjeong/research:human_fitting
$ conda activate py37
```

#### 1. Mesh rendering
docker: jbjeong/research:human_fitting
```bash
$ cd {home}/human_fitting
$ python mesh_rendering.py --mesh_dir mesh_data --out_dir data_root
```
Output -> {depth, images, pcd.ply, pcd_normal.npy}

#### 2. Skin detection
[code from: https://github.com/Jeanvit/PySkinDetection]
docker: jbjeong/research:human_fitting
```bash
$ cd PySkinDetection/src
$ python main.py --dataset data_root
```
Output -> {skin}
```bash
$ cd {home}/human_fitting
$ python labeling_skin_point.py --dataset data_root
```
Output -> {pcd_body.ply, skin_label.npy}

#### 3. Pose estimation
docker: cwaffles/openpose-python
```bash
$ docker pull cwaffles/openpose-python
$ cd /openpose
$ cp {home}/human_fitting/openpose_run.py . 
$ python openpose_run.py --root {home}/human_fitting/data_root
```
Output -> {keypts}

#### 4. Pose triangulation
docker: jbjeong/research:human_fitting
```bash
$ cd {home}/human_fitting
$ python triangulation_mp.py --dataset data_root
```
Output -> {pose3d.npz}

#### 5. SMPL-X fitting
Download SMPL-X model. (https://github.com/vchoutas/smplx#downloading-the-model)
docker: jbjeong/research:human_fitting
```bash
$ cd {home}/human_fitting/smplx-fit
$ python fit_smplx_torch.py --root /root/code/human_fitting/data_root
```
Output -> {
* result_step0.obj
* result_step1.obj
* result_step2.obj
* result_step3.obj
* result_step4.obj
* result_step5.obj
* result_step6.obj
* result_step7.obj
* result_step_face.obj
* result_step_fingers.obj
* result_step_wrist.obj
* output.obj
* output_aux.npz
}
