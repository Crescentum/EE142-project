# CelebA Training Commands

这些命令只用于你的 CelebA 部分。不要用公共的 `train.py`、`trainer.py` 或 `trainer_v.py` 跑 CelebA 实验。

## 1. 从头训练当前 CelebA InfoGAN

```powershell
python train_celeba.py --mode vanilla --epochs 10 --updates_per_epoch 200 --batch_size 128 --lambda_disc 1.0 --lambda_cont 0 --n_critic 1 --lr_g 0.0002 --lr_d 0.0002 --ckpt_dir checkpoints\celeba_stage4_single_code_infogan --log_dir logs\celeba_stage4_single_code_infogan
```

## 2. 从已有 checkpoint 继续训到 50 epoch

```powershell
python train_celeba.py --mode vanilla --epochs 50 --updates_per_epoch 200 --batch_size 128 --resume checkpoints\celeba_stage4_single_code_infogan\celeba_vanilla_final.pt --ckpt_dir checkpoints\celeba_stage4_single_code_infogan --log_dir logs\celeba_stage4_single_code_infogan
```

## 3. 如果 50 epoch 还在变好，继续训到 100 epoch

```powershell
python train_celeba.py --mode vanilla --epochs 100 --updates_per_epoch 200 --batch_size 128 --resume checkpoints\celeba_stage4_single_code_infogan\celeba_vanilla_final.pt --ckpt_dir checkpoints\celeba_stage4_single_code_infogan --log_dir logs\celeba_stage4_single_code_infogan
```

## 4. 生成 CelebA latent traversal 图片

默认会读取 `checkpoints\celeba_stage4_single_code_infogan\celeba_vanilla_final.pt`。

```powershell
python visualize_celeba.py
```

指定 checkpoint:

```powershell
python visualize_celeba.py --ckpt checkpoints\celeba_stage4_single_code_infogan\celeba_vanilla_final.pt --out_dir results\celeba_stage4_single_code_infogan_from_ckpt
```

输出图片：

```text
results\celeba_stage4_single_code_infogan_from_ckpt\celeba_vanilla_figure2.png
```

## 5. 导出 CelebA loss 曲线和 TensorBoard 图片

默认会读取 `logs\celeba_stage4_single_code_infogan`。

```powershell
python export_results_celeba.py
```

指定路径:

```powershell
python export_results_celeba.py --log_dir logs\celeba_stage4_single_code_infogan --out_dir results\celeba_stage4_single_code_infogan_exported
```

关键输出：

```text
results\celeba_stage4_single_code_infogan_exported\summary.md
results\celeba_stage4_single_code_infogan_exported\...\loss_curves.png
results\celeba_stage4_single_code_infogan_exported\...\traversal_c1_category_step*.png
```

## 6. 当前实验设置说明

- CelebA 使用 `model_celeba.py`
- CelebA 专用训练入口是 `train_celeba.py`
- CelebA 专用 trainer 是 `trainer_celeba.py`
- CelebA 专用可视化脚本是 `visualize_celeba.py`
- CelebA 专用导出脚本是 `export_results_celeba.py`
- 当前版本先使用 1 个 categorical latent code，每个 code 有 10 类
- 先证明能生成脸，再证明 `mi_disc` 下降、latent traversal 有变化
