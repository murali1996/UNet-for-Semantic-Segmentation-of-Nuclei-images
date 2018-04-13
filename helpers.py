# -*- coding: utf-8 -*-

import cv2, numpy as np, os
import params
from keras.preprocessing import image as kimage

######################### RLE TO MASK ###########################################################
def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

######################### IMAGE UNET INPUT ###########################################################
def append_hsv_image(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV);
    image = np.concatenate((bgr_image, hsv_image), axis=-1).astype(float);
    return image;
def mean_substraction(image):
    for channel in range(image.shape[-1]):
        image[:,:,channel]-=np.mean(image[:,:,channel]);
    return image;
def prep_image_to_unet(image_path, image): # image = kimage.load_img(image_path); image = kimage.img_to_array(image); pimage = preprocess_input(image);
    if image_path is None:
        bgr_image = image.copy();
    elif image is None:
        bgr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED);
    image = append_hsv_image(bgr_image);
    image = mean_substraction(image);
    return image
def prep_mask_to_unet(mask_path):
    mask = kimage.load_img(mask_path)
    mask = kimage.img_to_array(mask)
    mask = mask[:,:,:1]/255; # Reads like RGB Image. Hence we take only one frame
    return mask

######################### IMAGE AUGMENTATION ###########################################################
def generate_random_rotations_by90s(image, bw_mask):  # Images Rotated as angle%90 degrees only!    # Ex: image shape: 234*125*3, bw_mask shape: 256*256
    ret_angles = [90*int(angle/90) for angle in params.rotate_angles];
    r_image_generated, r_bw_mask_generated = [], [];
    if image is not None:
        for angle in ret_angles:
            n90_times = int(angle/90);
            r_image = [];
            for channel in range(image.shape[-1]):
                thisChannel = image[:,:,channel].copy()
                for _ in range(n90_times):
                    thisChannel = np.rot90(thisChannel);
                r_image.append(thisChannel);
            r_image = np.stack(r_image, axis=-1)
            r_image_generated.append(r_image)
    if bw_mask is not None:
        for angle in ret_angles:
            n90_times = int(angle/90);
            r_bw_mask = [];
            for channel in range(1):
                thisChannel = bw_mask[:,:].copy()
                for _ in range(n90_times):
                    thisChannel = np.rot90(thisChannel);
                r_bw_mask.append(thisChannel);
            r_bw_mask = np.stack(r_bw_mask, axis=-1)
            r_bw_mask_generated.append(r_bw_mask)
    return ret_angles, r_image_generated, r_bw_mask_generated;
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=1):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image
############################## TRAIN DATA STATS #########################################################
# Get sizes of training images and testing images
def analyze_data():
    tr_shapes = [];
    tr_shapes_not_256x256 = 0;
    for item in os.listdir(params.train_folder_org):
        # Get Paths and # Load image
        image_path = os.path.join(params.train_folder_org,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        tr_shapes+=[[image.shape[0],image.shape[1]]];
        if image.shape[0]!=256 and image.shape[1]!=256:
            tr_shapes_not_256x256+=1
    ts_shapes = [];
    ts_shapes_not_256x256 = 0;
    for item in os.listdir(params.test_folder_org):
        # Get Paths and # Load image
        image_path = os.path.join(params.test_folder_org,item+'/images');
        image_file = os.listdir(image_path)[0];
        image = cv2.imread(os.path.join(image_path,image_file));
        ts_shapes+=[[image.shape[0],image.shape[1]]];
        if image.shape[0]!=256 and image.shape[1]!=256:
            ts_shapes_not_256x256+=1
    return tr_shapes, tr_shapes_not_256x256, ts_shapes, ts_shapes_not_256x256

############################## RLE #########################################################
def rle_encoding(x, label=1):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==label)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = [str(i) for i in run_lengths];
    #return run_lengths
    return ' '.join(run_lengths)

def make_patches_and_predict(model, bgr_image, threshold_type='otsu', print_images=False): #'global','otsu','adap+otsu'
    # Create patch->Predict->Save in Mask->Increment each_pixel_cnt for resp. patch
    # Loading image and Initialization
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

