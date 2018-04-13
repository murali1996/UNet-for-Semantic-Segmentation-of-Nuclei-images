# -*- coding: utf-8 -*-
import os, random

############################################# DATA ANALYSIS ###################################################
hsv_clustering_data = os.path.join(os.curdir, "hsv_clustering.pickle");

############################################# DATA AUGMENTATION ###################################################
max_non_mask_pixel_percent = 0.95
win_shift_rows, win_shift_cols = 128, 128;
image_patch_rows, image_patch_cols, image_patch_channels = 256, 256, 3; # shape_dim 1 and 2 MUST be multiples of 16
win_shift_rows_test, win_shift_cols_test = 64, 64;
image_patch_rows_test, image_patch_cols_test, image_patch_channels_test = 256, 256, 3; # shape_dim 1 and 2 MUST be multiples of 16
rotate_angles = [0]+random.sample([90, 180, 270],1) #In degrees

############################################# MAIN FOLDERS ###################################################
data_dummy = os.path.join(os.pardir,r'data_dummy');

train_folder_org  = os.path.join(os.pardir,'data_org/stage1_train');
train_folder_gen   = os.path.join(os.pardir,'data_gen/train');
train_folder_gen_resize   = os.path.join(os.pardir,'data_gen/train_resize');

test_folder_org   = os.path.join(os.pardir,'data_org/stage1_test');
test_folder_gen   = os.path.join(os.pardir,'data_gen/test1');

test2_folder_org   = os.path.join(os.pardir,'data_org/stage2_test_final');
test2_folder_gen   = os.path.join(os.pardir,'data_gen/test2');

train_folder_gen_m1  = os.path.join(os.pardir,'data_gen/m1/train');
train_folder_gen_m2  = os.path.join(os.pardir,'data_gen/m2/train');
train_folder_gen_m3  = os.path.join(os.pardir,'data_gen/m3/train');

############################################# MNODEL TRAINING ###################################################
unet_input = (None, None, 3); #RGB+HSV
batch_size = 4; #Initially 16
n_epochs = 100;
base_model_weights_train = os.path.join(os.pardir, "model_weights/m3/train")
base_model_weights_validate = os.path.join(os.pardir, "model_weights/m3/validate")
logs_folder = os.path.join(os.pardir, "logs/m3")
############################################# MODEL TESTING ###################################################
best_model_path_m1 = os.path.join(base_model_weights_validate,r'weights-epoch-13-loss-0.1048-val_loss-0.1183.hdf5')
best_model_path_m2 = os.path.join(base_model_weights_validate,r'weights-epoch-20-loss-0.1434-val_loss-0.1685.hdf5')
best_model_path_m3 = os.path.join(base_model_weights_validate,r'weights-epoch-35-loss-0.1771-val_loss-0.1767.hdf5');
submit_path = os.path.join(os.pardir,r'submissions');

############################################# MISC. ###################################################
seed = 17;
smooth = 1.;

######################################## CLUSTER FILES ########################################################
#clusters_folder_m3 = os.path.join(os.pardir, 'data_gen/m3/clusters');
#
#cluster_0_folder_org = os.path.join(os.pardir, 'data_gen/m3/clusters/0');
#cluster_1_folder_org = os.path.join(os.pardir, 'data_gen/m3/clusters/1');
#cluster_2_folder_org = os.path.join(os.pardir, 'data_gen/m3/clusters/2');
#cluster_3_folder_org = os.path.join(os.pardir, 'data_gen/m3/clusters/3');
#
#cluster_0_folder_aug = os.path.join(os.pardir, 'data_gen/m3/clusters/0_aug');
#cluster_1_folder_aug = os.path.join(os.pardir, 'data_gen/m3/clusters/1_aug');
#cluster_2_folder_aug = os.path.join(os.pardir, 'data_gen/m3/clusters/2_aug');
#cluster_3_folder_aug = os.path.join(os.pardir, 'data_gen/m3/clusters/3_aug');
#
#base_model_weights_train_0 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/train/0');
#base_model_weights_validate_0 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/validate/0');
#logs_folder_0 = os.path.join(os.pardir, 'data_gen/m3/clusters/logs/0');
#base_model_weights_train_1 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/train/1');
#base_model_weights_validate_1 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/validate/1');
#logs_folder_1 = os.path.join(os.pardir, 'data_gen/m3/clusters/logs/1');
#base_model_weights_train_2 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/train/2');
#base_model_weights_validate_2 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/validate/2');
#logs_folder_2 = os.path.join(os.pardir, 'data_gen/m3/clusters/logs/2');
#base_model_weights_train_3 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/train/3');
#base_model_weights_validate_3 = os.path.join(os.pardir, 'data_gen/m3/clusters/model_weights/validate/3');
#logs_folder_3 = os.path.join(os.pardir, 'data_gen/m3/clusters/logs/3');
#
#
#best_model_path_m3_c0 = os.path.join(base_model_weights_validate_0,
#                                     r'weights-epoch-23-loss-0.0709-val_loss-0.1210.hdf5')
#best_model_path_m3_c1 = os.path.join(base_model_weights_validate_1,
#                                     r'weights-epoch-86-loss-0.0616-val_loss-0.1704.hdf5')
#best_model_path_m3_c2 = best_model_path_m3
#best_model_path_m3_c3 = os.path.join(base_model_weights_validate_3,
#                                     r'weights-epoch-64-loss-0.0926-val_loss-0.2117.hdf5')
#
#color_transfer_target_label3 = r'';
#color_transfer_target_label1 = r'';
