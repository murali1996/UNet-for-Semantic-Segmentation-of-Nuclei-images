# -*- coding: utf-8 -*-
# NOTES
# ALl mask related mathematics happens assuming (and only after conversion to) the mask image as a 2D array [... x ...]
# No channels=1 are used for the mask. Ofcourse while reading, it is read as [.. x .. x 1];

import os, cv2, pickle as pk, numpy as np
import params
from clustering import dominant_clusters

###########################################################################################################
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
    ###############################################################
    # Collects all masks from the org folder and erodes any two
    # masks thata re close by.
    ###############################################################
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
#############################################################################################################
def thisPatch(image, bw_mask, curr_row, curr_col, base_name, vertical_flip, horizontal_flip, rotation_90s, zoomIn):
    # Desired values
    row_length, col_length, channels = params.image_patch_rows, params.image_patch_cols, params.image_patch_channels;
    # make new images
    new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
    new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length].copy();
    new_name = base_name;
    # Random Operations
    if vertical_flip and np.random.uniform(0,1)>=0.5:
        new_image = np.flip(new_image,axis=0);
        new_mask = np.flip(new_mask,axis=0);
        new_name = new_name+'_vflip';
    if horizontal_flip and np.random.uniform(0,1)>=0.5:
        new_image = np.flip(new_image,axis=1);
        new_mask = np.flip(new_mask,axis=1);
        new_name = new_name+'_hflip';
    if rotation_90s and np.random.uniform(0,1)>=0.5 and new_image.shape[0]==new_image.shape[1]:
        # Using OpenCV, input must be SQUARE else bizzare results can be seen
        rows,cols,_ = new_image.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90*(np.random.randint(3)+1),1)
        new_image = cv2.warpAffine(new_image,M,(cols,rows))
        new_mask = cv2.warpAffine(new_mask,M,(cols,rows))
        new_name = new_name+'_r90';
    if zoomIn and np.random.uniform(0,1)>=0.5:
        # Up-Size the image and do rotation and capture 256*256 size
        new_image_big = cv2.resize(new_image, (2*new_image.shape[1], 2*new_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        new_mask_big = cv2.resize(new_mask, (2*new_mask.shape[1], 2*new_mask.shape[0]), interpolation = cv2.INTER_LINEAR)
        start_row = np.random.randint(new_image_big.shape[0]-row_length)
        start_col = np.random.randint(new_image_big.shape[1]-col_length)
        new_image = new_image_big[start_row:start_row+row_length,start_col:start_col+col_length,:];
        new_mask = new_mask_big[start_row:start_row+row_length,start_col:start_col+col_length];
        new_name = new_name+'_zoomin';
    return new_image, new_mask, new_name
def extract_patches_and_save(item, image, bw_mask, vertical_flip, horizontal_flip, rotation_90s, zoomIn):
    # Desired values
    row_length, col_length = params.image_patch_rows, params.image_patch_cols;
    # Other Initializations
    extracted_images, extracted_masks, extracted_names = [], [], [];
    img_rows, img_cols, img_channels = image.shape;
    # raise Exception('Channels<Desired Channels') if(img_channels<channels) else print('Shapes of Input Read: ', image.shape, bw_mask.shape)
    # Generate
    curr_gen_count=1;
    curr_row = 0;
    while(curr_row+row_length<=img_rows):
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            base_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
            new_image, new_mask, new_gen_name = thisPatch(image, bw_mask, curr_row, curr_col, base_name, vertical_flip, horizontal_flip, rotation_90s, zoomIn);
            if (np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]))<params.max_non_mask_pixel_percent or len(extracted_names)<1:
                extracted_images.append(new_image); extracted_masks.append(new_mask); extracted_names.append(new_gen_name);
                curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            base_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
            new_image, new_mask, new_gen_name = thisPatch(image, bw_mask, curr_row, curr_col, base_name, vertical_flip, horizontal_flip, rotation_90s, zoomIn);
            if (np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]))<params.max_non_mask_pixel_percent or len(extracted_names)<1:
                extracted_images.append(new_image); extracted_masks.append(new_mask); extracted_names.append(new_gen_name);
                curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
        curr_row+=params.win_shift_rows;
    # Saving a last patch in the given set of varying row number and varying col number
    if curr_row>0 and curr_row<img_rows and curr_row+row_length>img_rows and (img_rows-row_length)%params.win_shift_rows!=0: #Dont go to one of earlier indices
        curr_row = img_rows-row_length;
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            base_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
            new_image, new_mask, new_gen_name = thisPatch(image, bw_mask, curr_row, curr_col, base_name, vertical_flip, horizontal_flip, rotation_90s, zoomIn);
            if (np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]))<params.max_non_mask_pixel_percent or len(extracted_names)<1:
                extracted_images.append(new_image); extracted_masks.append(new_mask); extracted_names.append(new_gen_name);
                curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            base_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
            new_image, new_mask, new_gen_name = thisPatch(image, bw_mask, curr_row, curr_col, base_name, vertical_flip, horizontal_flip, rotation_90s, zoomIn);
            if (np.sum(new_mask==0)/(new_mask.shape[0]*new_mask.shape[1]))<params.max_non_mask_pixel_percent or len(extracted_names)<1:
                extracted_images.append(new_image); extracted_masks.append(new_mask); extracted_names.append(new_gen_name);
                curr_gen_count+=1;
            curr_col+=params.win_shift_cols;
    return extracted_images, extracted_masks, extracted_names
#############################################################################################################
def generate_train_data_with_augmentation(target_folder): # Patches with FLIPS, ROTATIONS, ZOOMIN
    # Sanity check for files of requirement
    with open(params.hsv_clustering_data,'rb') as openFile:
        hsv_clustering = pk.load(openFile);
        clt_model = hsv_clustering['clt']
        print("HSV Cluster Centers: ", np.array(hsv_clustering['cluster_centers_'], dtype='int'));
        del hsv_clustering;
    # Other initializations
    org_names, gen_names, gen_names_ind, gen_names_ind_cnt = [], [], [], []; #gen_names_individual
    for item in os.listdir(params.train_folder_org):
        # Get Paths & Load image,masks
        org_names+=[item]; print(len(org_names));
        image_org_path = os.path.join(params.train_folder_org,item+'/images');
        mask_org_path = os.path.join(params.train_folder_org,item+'/masks');
        image = cv2.imread(os.path.join(image_org_path,os.listdir(image_org_path)[0]));
        bw_mask = get_bw_mask(mask_org_path);
        # Know to which clustre this image belongs to!
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV);
        thisCenter, _, _ = dominant_clusters(hsv_image, n_clusters=1);
        centerMapped = clt_model.predict(thisCenter);
        n_iters = 1; vertical_flip=True; horizontal_flip=True; rotation_90s=True; zoomIn=False;
        if centerMapped==1 or centerMapped==3:
            n_iters = 4; zoomIn=True;
        for _ in range(n_iters):
            # Patches with FLIPS, ROTATIONS, ZOOMIN
            extracted_images, extracted_masks, extracted_names = extract_patches_and_save(item, image, bw_mask,
                                                                                          vertical_flip, horizontal_flip, rotation_90s, zoomIn);
            # Write images to folder of interest
            print('No.of Files Generated: {}'.format(len(extracted_names)))
            for ind, name in enumerate(extracted_names):
                cv2.imwrite(os.path.join(target_folder, name+'.png'), extracted_images[ind]);
                cv2.imwrite(os.path.join(target_folder, name+'_mask.png'), extracted_masks[ind]);
            # Save data for future reference and proceed
            gen_names+=extracted_names;
            gen_names_ind+=[extracted_names];
            gen_names_ind_cnt+=[len(extracted_names)];
    with open(os.path.join(target_folder,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    with open(os.path.join(target_folder,'gen_names.pickle'),'wb') as opfile:
        pk.dump(gen_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names, gen_names, gen_names_ind, gen_names_ind_cnt
def generate_train_data_only_resize(target_folder): # Rotations+Patches
    org_names, gen_names = [], []; #gen_names_individual
    for item in os.listdir(params.train_folder_org):
        # Get Paths & Load image,masks
        org_names+=[item]; print(len(org_names));
        image_org_path = os.path.join(params.train_folder_org,item+'/images');
        mask_org_path = os.path.join(params.train_folder_org,item+'/masks');
        image = cv2.imread(os.path.join(image_org_path,os.listdir(image_org_path)[0]));
        bw_mask = get_bw_mask(mask_org_path);
        # Desired values
        row_length, col_length = params.image_patch_rows, params.image_patch_cols;
        # Resize to desired size
        image = cv2.resize(image, (col_length, row_length), interpolation = cv2.INTER_LINEAR);
        bw_mask = cv2.resize(bw_mask, (col_length, row_length), interpolation = cv2.INTER_LINEAR);
        for _ in range(2):
            # Do FLIPS, ROTATIONS, ZOOMIN
            new_image, new_mask, new_name = thisPatch(image, bw_mask, 0, 0, item);
            # Write images to folder of interest
            cv2.imwrite(os.path.join(target_folder, new_name+'.png'), new_image);
            cv2.imwrite(os.path.join(target_folder, new_name+'_mask.png'), new_mask);
            # Save data for future reference and proceed
            gen_names+=[new_name];
    with open(os.path.join(target_folder,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    with open(os.path.join(target_folder,'gen_names.pickle'),'wb') as opfile:
        pk.dump(gen_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names, gen_names
############################################################################################################
def load_train_list_with_augmentation(target_folder):
    with open(os.path.join(target_folder,'gen_names.pickle'),'rb') as opfile:
        tr_list = pk.load(opfile);
        opfile.close();
    return tr_list
def load_test_list(target_folder):
    with open(os.path.join(target_folder,'org_names.pickle'),'rb') as opfile:
        ts_list = pk.load(opfile);
        opfile.close();
    return ts_list
############################################################################################################
def generate_original_copies_train_data(source_folder, target_folder):
    org_names = [];
    for item in os.listdir(source_folder):
        org_names+=[item]; print(len(org_names));
        # Get Path and load image
        image_path = os.path.join(source_folder,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        cv2.imwrite(os.path.join(target_folder, item+'.png'), image);
    with open(os.path.join(target_folder,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names
def generate_original_copies_test_data(source_folder, target_folder):
    org_names = [];
    for item in os.listdir(source_folder):
        org_names+=[item]; print(len(org_names));
        # Get Path and load image
        image_path = os.path.join(source_folder,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        cv2.imwrite(os.path.join(target_folder, item+'.png'), image);
    with open(os.path.join(target_folder,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names
############################################################################################################
#from train_m3 import prep_image_to_unet
#def make_patches_and_predict(model, bgr_image, threshold_type='global', print_images=False): #'global','otsu','adap+otsu'
#    # Create patch->Predict->Save in Mask->Increment each_pixel_cnt for resp. patch
#    img_rows, img_cols, channels = bgr_image.shape;
#    row_length, col_length = params.image_patch_rows, params.image_patch_cols; # Desired values
#    cnt_matrix = np.zeros((bgr_image.shape[0],bgr_image.shape[1],1));
#    mask_matrix = np.zeros((bgr_image.shape[0],bgr_image.shape[1],1));
#    if(channels<3):
#        raise Exception('Channels<Desired Channels')
#    curr_gen_count =  1;
#    #Patches
#    curr_row = 0;
#    while(curr_row+row_length<=img_rows):
#        curr_col = 0;
#        while(curr_col+col_length<=img_cols):
#            # newImage and predict\
#            #new_bgr_image = bgr_image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
#            new_bgr_image = bgr_image;
#            new_bgr_image, new_ycrcb_image = prep_image_to_unet(None, new_bgr_image);
#            new_bgr_image = np.expand_dims(new_bgr_image, axis=0);
#            new_ycrcb_image = np.expand_dims(new_ycrcb_image, axis=0)
#            pred_mask = model.predict([new_ycrcb_image,new_bgr_image])[0];
#            # save details
#            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
#            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
#            # Increment count and continue
#            curr_gen_count+=1;
#            curr_col+=params.win_shift_cols;
#        # Saving a last patch in the given set of fixed rows and varying col number
#        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
#            curr_col = img_cols-col_length;
#            # newImage and predict\
#            new_bgr_image = bgr_image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
#            new_bgr_image, new_ycrcb_image = prep_image_to_unet(None, new_bgr_image);
#            new_bgr_image = np.expand_dims(new_bgr_image, axis=0);
#            new_ycrcb_image = np.expand_dims(new_ycrcb_image, axis=0)
#            pred_mask = model.predict([new_ycrcb_image,new_bgr_image])[0];
#            # save details
#            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
#            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
#            # Increment count and continue
#            curr_gen_count+=1;
#        curr_row+=params.win_shift_rows;
#    # Saving a last patch in the given set of varying row number and varying col number
#    if curr_row>0 and curr_row<img_rows and curr_row+row_length>img_rows and (img_rows-row_length)%params.win_shift_rows!=0: #Dont go to one of earlier indices
#        curr_row = img_rows-row_length;
#        curr_col = 0;
#        while(curr_col+col_length<=img_cols):
#            # newImage and predict\
#            new_bgr_image = bgr_image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
#            new_bgr_image, new_ycrcb_image = prep_image_to_unet(None, new_bgr_image);
#            new_bgr_image = np.expand_dims(new_bgr_image, axis=0);
#            new_ycrcb_image = np.expand_dims(new_ycrcb_image, axis=0)
#            pred_mask = model.predict([new_ycrcb_image,new_bgr_image])[0];
#            # save details
#            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
#            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
#            # Increment count and continue
#            curr_gen_count+=1;
#            curr_col+=params.win_shift_cols;
#        # Saving a last patch in the given set of fixed rows and varying col number
#        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
#            curr_col = img_cols-col_length;
#            # newImage and predict\
#            new_bgr_image = bgr_image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels].copy();
#            new_bgr_image, new_ycrcb_image = prep_image_to_unet(None, new_bgr_image);
#            new_bgr_image = np.expand_dims(new_bgr_image, axis=0);
#            new_ycrcb_image = np.expand_dims(new_ycrcb_image, axis=0)
#            pred_mask = model.predict([new_ycrcb_image,new_bgr_image])[0];
#            # save details
#            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
#            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
#            # Increment count and continue
#            curr_gen_count+=1;
#    if np.min(cnt_matrix)==0: # Safe Check
#        raise Exception('At least one pixel is not in any of the patches! Use np.where() to find it.')
#    # Threshold the pixels
#    mask_matrix/=cnt_matrix; # Average ##### mask_matrix = (mask_matrix > 0.5).astype(np.uint8) ######
#    if threshold_type=='global':
#        temp = mask_matrix.copy();
#        pred_image_thres = 255*(temp > 0.5);
#        pred_image_thres = pred_image_thres.astype(np.uint8);
#    elif threshold_type=='otsu':
#        temp = mask_matrix.copy();
#        pred_image_255 = 255*temp;
#        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
#        _ , pred_image_thres = cv2.threshold(pred_image_255_uint8, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    elif threshold_type=='adap+otsu':
#        #OTSU
#        temp = mask_matrix.copy();
#        pred_image_255 = 255*temp;
#        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
#        _ , pred_image_thres1 = cv2.threshold(pred_image_255_uint8, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        # Adaptive Thresholding
#        temp = mask_matrix.copy();
#        pred_image_255 = 255*temp;
#        pred_image_255_uint8 = pred_image_255.astype(np.uint8)
#        pred_image_thres2 = cv2.adaptiveThreshold(pred_image_255_uint8, 255,
#                                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                  thresholdType=cv2.THRESH_BINARY, blockSize=13, C=0)
#        # Global Thresholding AND Adaptive Thresholding
#        pred_image_thres = np.logical_or(pred_image_thres1==255,pred_image_thres2==255);
#        pred_image_thres = pred_image_thres.astype(np.uint8)
#        pred_image_thres = 255*pred_image_thres;
#    else:
#        print('Required threshold_type unavailable!');
#    # Print thye images
#    if print_images:
#        cv2.imshow( "{0}".format('ThisImage'), bgr_image);
#        cv2.imshow( "{0}".format('ThisMask'), pred_image_thres);
#        cv2.waitKey(0);
#        cv2.destroyAllWindows();
#    return pred_image_thres #, mask_matrix

if __name__=="__main__":
    generate_original_copies_train_data(params.train_folder_org, params.train_folder_gen);
    #generate_original_copies_test_data(params.test_folder_org, params.test_folder_gen);
    #generate_original_copies_test_data(params.test2_folder_org, params.test2_folder_gen);
    #org_names, gen_names, gen_names_ind, gen_names_ind_cnt = generate_train_data_with_augmentation(params.train_folder_gen_m3); #Uses global variable
    #org_names, gen_names = generate_train_data_only_resize(params.train_folder_gen_resize)
    #tr_list = load_train_list_with_augmentation(params.train_folder_gen);
    #ts_list = load_test_list(params.test_folder_gen)
    print('Execution Complete. Please check respective folders as mentioned in params.')