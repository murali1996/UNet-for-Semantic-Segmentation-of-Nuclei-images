# -*- coding: utf-8 -*-
# NOTES
# ALl mask related mathematics happens assuming (and only after conversion to) the mask image as a 2D array [... x ...]
# No channels=1 are used for the mask. Ofcourse while reading, it is read as [.. x .. x 1];

import os, cv2, pickle as pk, numpy as np, pickle
import params
from helpers import generate_rotations_by90s
from train_m2_clst import prep_image_to_unet

###########################################################################################################
def save_generated_rotations(name_head, r_angles, r_images_generated, r_bw_mask_generated, target_folder):
    generated_names = [];
    for ind, angle in enumerate(r_angles): # Save rotated Image
        gen_name = str(name_head)+'_'+str(angle);
        generated_names+=[gen_name]
        cv2.imwrite(os.path.join(target_folder, gen_name+'.png'), r_images_generated[ind]);
        if r_bw_mask_generated!=[]:
            cv2.imwrite(os.path.join(target_folder, gen_name+'_mask.png'), r_bw_mask_generated[ind]);
    return generated_names
def gen_rotations_and_save(target_folder, name_head, image, bw_mask=None):
    r_angles, r_images_generated, r_bw_mask_generated = generate_rotations_by90s(image, bw_mask);
    return save_generated_rotations(name_head, r_angles, r_images_generated, r_bw_mask_generated, target_folder);
def generate_train_data_augmentation(source_folder = params.cluster_1_folder_org,
                                     target_folder = params.cluster_1_folder_aug,
                                     ):
    with open(os.path.join(source_folder,'org_names.pickle'), 'rb') as opfile:
        image_files = pickle.load(opfile); opfile.close();
    row_length, col_length, channels = params.image_patch_rows, params.image_patch_cols, params.image_patch_channels; # Desired values
    org_names, gen_names, gen_names_ind, gen_names_ind_cnt = [], [], [], []; #gen_names_individual
    for item in image_files:
        # Keep a track of items
        org_names+=[item]; print(len(org_names));
        # Get Paths
        image_path = os.path.join(source_folder,item+'.png');
        mask_path = os.path.join(source_folder,item+'_mask.png');
        # Load image and masks
        image = cv2.imread(image_path);
        bw_mask = cv2.imread(mask_path)[:,:,0];
        print(image.shape, bw_mask.shape);
        # Generate
        img_rows, img_cols, img_channels = image.shape;
        if(img_channels<channels):
            raise Exception('Channels<Desired Channels')
        curr_gen_names, curr_gen_count, curr_row, curr_col = [], 1, 0, 0;
        curr_row = 0;
        while(curr_row+row_length<=img_rows):
            curr_col = 0;
            while(curr_col+col_length<=img_cols):
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Consider patch only if % of background pixels<=80%
                if np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]) < params.max_non_mask_pixel_percent or len(curr_gen_names)<1:
                    new_names = gen_rotations_and_save(target_folder, new_gen_name, new_image, new_mask)
                    curr_gen_names+=new_names;
                    curr_gen_count+=1; # Continue iterations
                curr_col+=params.win_shift_cols;
            # Saving a last patch in the given set of fixed rows and varying col number
            if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
                curr_col = img_cols-col_length;
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Consider patch only if % of background pixels<=80%
                if np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]) < params.max_non_mask_pixel_percent or len(curr_gen_names)<1:
                    new_names = gen_rotations_and_save(target_folder, new_gen_name, new_image, new_mask)
                    curr_gen_names+=new_names;
                    curr_gen_count+=1; # Continue iterations
                curr_col+=params.win_shift_cols;
            curr_row+=params.win_shift_rows;
        # Saving a last patch in the given set of varying row number and varying col number
        if curr_row>0 and curr_row<img_rows and curr_row+row_length>img_rows and (img_rows-row_length)%params.win_shift_rows!=0: #Dont go to one of earlier indices
            curr_row = img_rows-row_length;
            curr_col = 0;
            while(curr_col+col_length<=img_cols):
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Consider patch only if % of background pixels<=80%
                if np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]) < params.max_non_mask_pixel_percent or len(curr_gen_names)<1:
                    new_names = gen_rotations_and_save(target_folder, new_gen_name, new_image, new_mask)
                    curr_gen_names+=new_names;
                    curr_gen_count+=1; # Continue iterations
                curr_col+=params.win_shift_cols;
                curr_col+=params.win_shift_cols;
            # Saving a last patch in the given set of fixed rows and varying col number
            if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
                curr_col = img_cols-col_length;
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Consider patch only if % of background pixels<=80%
                if np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]) < params.max_non_mask_pixel_percent or len(curr_gen_names)<1:
                    new_names = gen_rotations_and_save(target_folder, new_gen_name, new_image, new_mask)
                    curr_gen_names+=new_names;
                    curr_gen_count+=1; # Continue iterations
                curr_col+=params.win_shift_cols;
        gen_names+=curr_gen_names;
        gen_names_ind+=[curr_gen_names];
        gen_names_ind_cnt+=[len(curr_gen_names)];
    with open(os.path.join(target_folder,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    with open(os.path.join(target_folder,'gen_names.pickle'),'wb') as opfile:
        pk.dump(gen_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names, gen_names, gen_names_ind, gen_names_ind_cnt
############################################################################################################
#def generate_test_data():
#    org_names = [];
#    for item in os.listdir(params.test_folder_org):
#        org_names+=[item]; print(len(org_names));
#        # Get Path and load image
#        image_path = os.path.join(params.test_folder_org,item+'/images');
#        image_file = os.listdir(image_path)[0];
#        image = cv2.imread(os.path.join(image_path,image_file));
#        # Add dimensions
#        cv2.imwrite(os.path.join(params.test_folder_gen, item+'.png'), image);
#    with open(os.path.join(params.test_folder_gen,'org_names.pickle'),'wb') as opfile:
#        pk.dump(org_names, opfile); opfile.close();
#    print('Training Data generated success...');
#    return org_names
def load_test_list(cluster_folder):
    with open(os.path.join(params.test_folder_gen,'org_names.pickle'),'rb') as opfile:
        ts_list = pk.load(opfile);
        opfile.close();
    return ts_list

###########################################################################################################
def make_patches_and_predict(model, bgr_image, threshold_type='otsu', print_images=False): #'global','otsu','adap+otsu'
    # Create patch->Predict->Save in Mask->Increment each_pixel_cnt for resp. patch
    # Loading image and Initialization
#    image = prep_image_to_unet(None, bgr_image); # BGR+HSV Image; 6 dimensions
    image = bgr_image.copy();
    img_rows, img_cols, channels = image.shape;
    row_length, col_length = params.image_patch_rows, params.image_patch_cols; # Desired values
    cnt_matrix = np.zeros((image.shape[0],image.shape[1],1));
    mask_matrix = np.zeros((image.shape[0],image.shape[1],1));
    if(channels<3):
        raise Exception('Channels<Desired Channels')
    curr_gen_count, curr_row, curr_col =  1, 0, 0;
    #Patches
    curr_row = 0;
    while(curr_row+row_length<=img_rows):
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            # newImage and predict\
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
            new_image = prep_image_to_unet(None, new_image);
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            # make new images
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
            new_image = prep_image_to_unet(None, new_image);
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
        curr_row+=params.win_shift_rows;
    # Saving a last patch in the given set of varying row number and varying col number
    if curr_row>0 and curr_row<img_rows and curr_row+row_length>img_rows and (img_rows-row_length)%params.win_shift_rows!=0: #Dont go to one of earlier indices
        curr_row = img_rows-row_length;
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            # newImage and predict\
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
            new_image = prep_image_to_unet(None, new_image);
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            # make new images
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
            new_image = prep_image_to_unet(None, new_image);
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
    if np.min(cnt_matrix)==0: # Safe Check
        raise Exception('At least one pixel is not in any of the patches! Use np.where() to find it.')
    # Threshold the pixels
    mask_matrix/=cnt_matrix; # Average ##### mask_matrix = (mask_matrix > 0.5).astype(np.uint8) ######
    if threshold_type=='global':
        temp = mask_matrix.copy();
        pred_image_thres = 255*(temp > 0.5);
        pred_image_thres = pred_image_thres.astype(np.uint8);
    elif threshold_type=='otsu':
        temp = mask_matrix.copy();
        pred_image_255 = 255*temp;
        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
        _ , pred_image_thres = cv2.threshold(pred_image_255_uint8, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif threshold_type=='adap+otsu':
        #OTSU
        temp = mask_matrix.copy();
        pred_image_255 = 255*temp;
        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
        _ , pred_image_thres1 = cv2.threshold(pred_image_255_uint8, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Adaptive Thresholding
        temp = mask_matrix.copy();
        pred_image_255 = 255*temp;
        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
        pred_image_thres2 = cv2.adaptiveThreshold(pred_image_255_uint8, 255,
                                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  thresholdType=cv2.THRESH_BINARY, blockSize=13, C=0)
        # Global Thresholding AND Adaptive Thresholding
        pred_image_thres = np.logical_or(pred_image_thres1==255,pred_image_thres2==255);
        pred_image_thres = pred_image_thres.astype(np.uint8)
        pred_image_thres = 255*pred_image_thres;
    else:
        print('Required threshold_type unavailable!');
    # Print thye images
    if print_images:
        cv2.imshow( "{0}".format('ThisImage'), bgr_image);
        cv2.imshow( "{0}".format('ThisMask'), pred_image_thres);
        cv2.waitKey(0);
        cv2.destroyAllWindows();
    return pred_image_thres #, mask_matrix
############################################################################################################


if __name__=="__main__":
############################ Generate Augmented Data#############################################
#    source_folders = \
#        [params.cluster_1_folder_org, params.cluster_2_folder_org, params.cluster_3_folder_org]
#    target_folders = \
#        [params.cluster_1_folder_aug, params.cluster_2_folder_aug, params.cluster_3_folder_aug]
#    for ind, _ in enumerate(source_folders):
#        source_folder = source_folders[ind];
#        target_folder  = target_folders[ind];
#        if not os.path.exists(target_folder):
#            os.makedirs(target_folder);
#        _,_,_,_ = generate_train_data_augmentation(source_folder, target_folder);

#    source_folders = \
#        [params.cluster_0_folder_org]
#    target_folders = \
#        [params.cluster_0_folder_aug]
#    for ind, _ in enumerate(source_folders):
#        source_folder = source_folders[ind];
#        target_folder  = target_folders[ind];
#        if not os.path.exists(target_folder):
#            os.makedirs(target_folder);
#        _,_,_,_ = generate_train_data_augmentation(source_folder, target_folder);
    print('Nothing in store!!')