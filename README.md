# msfusion
Implementation of deep learning-based fusion method for combining multiple forensics masks, as described in the [A Multi-Stream Fusion Network for Image Splicing Localization](https://arxiv.org/abs/2212.01128.pdf) paper.

Example commands: python script.py training --experiment_name dct_sb --model_name dct_sb --batch_size 16 --gpu_id 1 --epochs 20 --checkpoint_path checkpoints/ --dataset_name synthetic

python evaluate_model.py evaluating --experiment_name dct --model_name dct --batch_size 16 --gpu_id 1 --checkpoint checkpoints/skip_connection_dct_casia/ckpt_11_3468.pth --dataset_name casia1


You can find the signals here: https://mever.iti.gr/msfusion/signals.zip
You can find the pretrained models here: https://mever.iti.gr/msfusion/checkpoints.zip

## Citations
If you use this code for your research, please consider citing our papers:
```bibtex
@inproceedings{siopi2023multi,
  title={A Multi-Stream Fusion Network for Image Splicing Localization},
  author={Siopi, Maria and Kordopatis-Zilos, Giorgos and Charitidis, Polychronis and Kompatsiaris, Ioannis and Papadopoulos, Symeon},
  booktitle={International Conference on Multimedia Modeling},
  year={2023}
}
```
