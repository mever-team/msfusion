# msfusion
Implementation of deep learning-based fusion method for combining multiple forensics masks.

Example commands: python script.py training --experiment_name dct_sb --model_name dct_sb --batch_size 16 --gpu_id 1 --epochs 20 --checkpoint_path checkpoints/ --dataset_name synthetic

python evaluate_model.py evaluating --experiment_name dct --model_name dct --batch_size 16 --gpu_id 1 --checkpoint checkpoints/skip_connection_dct_casia/ckpt_11_3468.pth --dataset_name casia1


You can find the signals here: https://mever.iti.gr/msfusion/signals.zip
You can find the pretrained models here: https://mever.iti.gr/msfusion/checkpoints.zip