python ../Frame_test.py \
--gpu_ids 4 \
--load_size 256 \
--crop_size 256 \
--model joint_gan \
--netG ShareSynNet \
--init_type kaiming \
--load_path /data/chwang/Log/JointFrame/joint_framework/8_net_G.pth