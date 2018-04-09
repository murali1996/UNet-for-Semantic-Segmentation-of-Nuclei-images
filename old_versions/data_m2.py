# -*- coding: utf-8 -*-
# NOTES
# ALl mask related mathematics happens assuming (and only after conversion to) the mask image as a 2D array [... x ...]
# No channels=1 are used for the mask. Ofcourse while reading, it is read as [.. x .. x 1];

import os, cv2, pickle as pk, numpy as np
import params
from helpers import generate_rotations_by90s

###########################################################################################################
def save_generated_rotations(name_head, r_angles, r_images_generated, r_bw_mask_generated):
    generated_names = [];
    for ind, angle in enumerate(r_angles): # Save rotated Image
        gen_name = str(name_head)+'_'+str(angle);
        generated_names+=[gen_name]
        cv2.imwrite(os.path.join(params.train_folder_gen_m2, gen_name+'.png'), r_images_generated[ind]);
        if r_bw_mask_generated!=[]:
            cv2.imwrite(os.path.join(params.train_folder_gen_m2, gen_name+'_mask.png'), r_bw_mask_generated[ind]);
    return generated_names
def gen_rotations_and_save(name_head, image, bw_mask=None):
    r_angles, r_images_generated, r_bw_mask_generated = generate_rotations_by90s(image, bw_mask);
    return save_generated_rotations(name_head, r_angles, r_images_generated, r_bw_mask_generated);
def simply_erode_opencv(inp_mask, adj_labels_to_erode):
    mask = inp_mask.copy();
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label in adj_labels_to_erode:
        # print(label)
        # Pick the partial_mask
        partial_mask = np.zeros(mask.shape); partial_mask[mask==label] = 1;
        mask[mask==label] = 0;
        er_partial_mask = cv2.erode(partial_mask, kernel, iterations = 2);
        mask[er_partial_mask==1] = label;
    return mask;
def get_bw_mask(mask_path, k_gap_erode=4):
    mask_files = os.listdir(mask_path);
    mask, partial_mask_label = [], 1;
    adj_labels_to_erode, k_gap = [], k_gap_erode;
    for mask_file in mask_files:
        partial_mask = cv2.imread(os.path.join(mask_path,mask_file), cv2.IMREAD_GRAYSCALE);
        partial_mask[partial_mask>0] = partial_mask_label;
        partial_mask_inds = np.where(partial_mask==partial_mask_label)
        try:
            mask+=partial_mask;
        except:
            mask = partial_mask;
        found_something_close_by = 0;
        for p in range(partial_mask_inds[0].shape[0]):
            row = partial_mask_inds[0][p]; col = partial_mask_inds[1][p];
            if row-k_gap>=0 and row+k_gap<partial_mask.shape[0] and col-k_gap>=0 and col+k_gap<partial_mask.shape[1]:
                window = mask[row-k_gap:row+k_gap, col-k_gap:col+k_gap];
                other_labels = window[(window>0) & (window!=partial_mask_label)]
                for label in other_labels:
                    if label not in adj_labels_to_erode:
                        #print(partial_mask_label, label, row, col);
                        adj_labels_to_erode.append(label);
                        found_something_close_by = 1;
        if found_something_close_by:
            #print(partial_mask_label);
            adj_labels_to_erode.append(partial_mask_label);
        # Continue adding more partial_masks
        partial_mask_label+=1;
    for ind, lab in enumerate(adj_labels_to_erode):
        adj_labels_to_erode[ind] = int(lab)
    # Simply erode these label clusters
    new_mask = simply_erode_opencv(mask, adj_labels_to_erode);
    # bw_mask =  cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY);
    bw_mask = new_mask.copy(); bw_mask[bw_mask>0] = 255;
    if len(adj_labels_to_erode)>0:
        print('These many masks were found closeby; Eroded Brutally!!', len(adj_labels_to_erode));
    return bw_mask
def generate_train_data_augmentation(): # Rotations+Patches
    row_length, col_length, channels = params.image_patch_rows, params.image_patch_cols, params.image_patch_channels; # Desired values
    org_names, gen_names, gen_names_ind, gen_names_ind_cnt = [], [], [], []; #gen_names_individual
    for item in os.listdir(params.train_folder_org):
        # Keep a track of items
        org_names+=[item]; print(len(org_names));
        # Get Paths
        image_path = os.path.join(params.train_folder_org,item+'/images');
        mask_path = os.path.join(params.train_folder_org,item+'/masks');
        # Load image and masks
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        bw_mask = get_bw_mask(mask_path);
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
                    new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask)
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
                    new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask)
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
                    new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask)
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
                    new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask)
                    curr_gen_names+=new_names;
                    curr_gen_count+=1; # Continue iterations
                curr_col+=params.win_shift_cols;
        gen_names+=curr_gen_names;
        gen_names_ind+=[curr_gen_names];
        gen_names_ind_cnt+=[len(curr_gen_names)];
    with open(os.path.join(params.train_folder_gen_m2,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    with open(os.path.join(params.train_folder_gen_m2,'gen_names.pickle'),'wb') as opfile:
        pk.dump(gen_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names, gen_names, gen_names_ind, gen_names_ind_cnt
def load_train_list():
    with open(os.path.join(params.train_folder_gen_m2,'gen_names.pickle'),'rb') as opfile:
        tr_list = pk.load(opfile);
        opfile.close();
    return tr_list
def generate_train_data():
    org_names = [];
    for item in os.listdir(params.train_folder_org):
        org_names+=[item]; print(len(org_names));
        # Get Path and load image
        image_path = os.path.join(params.train_folder_org,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        cv2.imwrite(os.path.join(params.train_folder_gen, item+'.png'), image);
    with open(os.path.join(params.train_folder_gen,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names
############################################################################################################
def generate_test_data():
    org_names = [];
    for item in os.listdir(params.test_folder_org):
        org_names+=[item]; print(len(org_names));
        # Get Path and load image
        image_path = os.path.join(params.test_folder_org,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        # Add dimensions
        cv2.imwrite(os.path.join(params.test_folder_gen, item+'.png'), image);
    with open(os.path.join(params.test_folder_gen,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names
def load_test_list():
    with open(os.path.join(params.test_folder_gen,'org_names.pickle'),'rb') as opfile:
        ts_list = pk.load(opfile);
        opfile.close();
    return ts_list

############################################################################################################
def make_patches_and_predict(model, pimage):
    from train_m2 import prep_image_to_unet
    image = pimage.copy();
    # Initialization for patch cretaion
    row_length, col_length, channels = params.image_patch_rows, params.image_patch_cols, params.image_patch_channels; # Desired values
    # Initialize Count matrix and mask_matrix
    cnt_matrix = np.zeros((image.shape[0],image.shape[1],1));
    mask_matrix = np.zeros((image.shape[0],image.shape[1],1));
    # Create patch->Get Predictions->Paste predictions->Increment each_pixel_cnt for resp. patch
    img_rows, img_cols, img_channels = image.shape;
    if(img_channels<3):
        raise Exception('Channels<Desired Channels')
    curr_gen_count, curr_row, curr_col =  1, 0, 0;
    #Patches
    curr_row = 0;
    while(curr_row+row_length<=img_rows):
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            # newImage and predict\
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = prep_image_to_unet(image=new_image);
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
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = prep_image_to_unet(image=new_image);
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
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = prep_image_to_unet(image=new_image);
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
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = prep_image_to_unet(image=new_image);
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;    # Safe Check
    if np.min(cnt_matrix)==0:
        raise Exception('At least one pixel is not in any of the patches! Use np.where() to find it.')
 ############################Under Construction!!#################################
 # Ideas, low pass filter, close-opening, before 0.5 thresholding!
 # Also, can win shift be just 16 or 32???
    # Threshold the pixels
    mask_matrix/=cnt_matrix;
    mask_matrix = (mask_matrix > 0.5).astype(np.uint8)
    # Pring thye images
    #cv2.imshow( "{0}".format('ThisMask'), 255*mask_matrix);
    #cv2.waitKey(0);
    #cv2.destroyAllWindows();
    return mask_matrix

############################################################################################################
if __name__=="__main__":
    # tr_shapes, tr_shapes_not_256x256, ts_shapes, ts_shapes_not_256x256 = analyze_data();
    org_names, gen_names, gen_names_ind, gen_names_ind_cnt = generate_train_data_augmentation();
    # test_list = generate_test_data();
    # tr_list = generate_train_data()
    print('Nothing in store!!')