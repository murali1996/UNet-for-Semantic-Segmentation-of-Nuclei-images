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
        cv2.imwrite(os.path.join(params.train_folder_gen_m1, gen_name+'.png'), r_images_generated[ind]);
        if r_bw_mask_generated!=[]:
            cv2.imwrite(os.path.join(params.train_folder_gen_m1, gen_name+'_mask.png'), r_bw_mask_generated[ind]);
    return generated_names
def gen_rotations_and_save(name_head, image, bw_mask=None):
    r_angles, r_images_generated, r_bw_mask_generated = generate_rotations_by90s(image, bw_mask);
    return save_generated_rotations(name_head, r_angles, r_images_generated, r_bw_mask_generated);
def generate_train_data_augmentation():
    # Making 4x data size: Save the data files in data_gen folder with Image Rotations
    # Also, data shape must be multiples of 16 because 4 times maxpool is being done
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
        mask_files = os.listdir(mask_path);
        mask = [];
        for mask_file in mask_files:
            partial_mask = cv2.imread(os.path.join(mask_path,mask_file));
            try:
                mask+=partial_mask;
            except:
                mask = partial_mask;
        bw_mask =  cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY);
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
                # Resize, rotate and save new names
                # new_image = resize_image_dims_multiple_of(16, new_image);
                # new_mask = resize_image_dims_multiple_of(16, np.expand_dims(new_mask, axis=-1));
                # new_mask = new_mask[:,:,0];
                # new_image, new_mask = resize_image_dims_multiple_of(16, image=new_image, bw_mask=new_mask)
                new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask, saveOriginal=True)
                curr_gen_names+=new_names;
                # Continue iterations
                curr_gen_count+=1;
                curr_col+=params.win_shift_cols;
            # Saving a last patch in the given set of fixed rows and varying col number
            if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
                curr_col = img_cols-col_length;
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Resize, rotate and save new names
                # new_image = resize_image_dims_multiple_of(16, new_image);
                # new_mask = resize_image_dims_multiple_of(16, np.expand_dims(new_mask, axis=-1));
                # new_mask = new_mask[:,:,0];
                # new_image, new_mask = resize_image_dims_multiple_of(16, image=new_image, bw_mask=new_mask)
                new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask, saveOriginal=True)
                curr_gen_names+=new_names;
                # Continue iterations
                curr_gen_count+=1;
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
                # Resize, rotate and save new names
                # new_image = resize_image_dims_multiple_of(16, new_image);
                # new_mask = resize_image_dims_multiple_of(16, np.expand_dims(new_mask, axis=-1));
                # new_mask = new_mask[:,:,0];
                # new_image, new_mask = resize_image_dims_multiple_of(16, image=new_image, bw_mask=new_mask)
                new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask, saveOriginal=True)
                curr_gen_names+=new_names;
                # Continue iterations
                curr_gen_count+=1;
                curr_col+=params.win_shift_cols;
            # Saving a last patch in the given set of fixed rows and varying col number
            if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols!=0: #Dont go to one of earlier indices
                curr_col = img_cols-col_length;
                # make new images
                new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
                new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
                new_gen_name = str(item)+'_'+str(row_length)+'x'+str(col_length)+'_'+str(curr_gen_count);
                # Resize, rotate and save new names
                # new_image = resize_image_dims_multiple_of(16, new_image);
                # new_mask = resize_image_dims_multiple_of(16, np.expand_dims(new_mask, axis=-1));
                # new_mask = new_mask[:,:,0];
                # new_image, new_mask = resize_image_dims_multiple_of(16, image=new_image, bw_mask=new_mask)
                new_names = gen_rotations_and_save(new_gen_name, new_image, new_mask, saveOriginal=True)
                curr_gen_names+=new_names;
                # Continue iterations
                curr_gen_count+=1;
        gen_names+=curr_gen_names;
        gen_names_ind+=[curr_gen_names];
        gen_names_ind_cnt+=[len(curr_gen_names)];
    with open(os.path.join(params.train_folder_gen_m1,'org_names.pickle'),'wb') as opfile:
        pk.dump(org_names, opfile); opfile.close();
    with open(os.path.join(params.train_folder_gen_m1,'gen_names.pickle'),'wb') as opfile:
        pk.dump(gen_names, opfile); opfile.close();
    print('Training Data generated success...');
    return org_names, gen_names, gen_names_ind, gen_names_ind_cnt
def load_train_list():
    with open(os.path.join(params.train_folder_gen_m1,'gen_names.pickle'),'rb') as opfile:
        tr_list = pk.load(opfile);
        opfile.close();
    return tr_list
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
    image = pimage.copy();
    # Initialization
    row_length, col_length, channels = params.image_patch_rows_test, params.image_patch_cols_test, params.image_patch_channels_test; # Desired values
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
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
            curr_col+=params.win_shift_cols_test;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols_test!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            # make new images
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
        curr_row+=params.win_shift_rows_test;
    # Saving a last patch in the given set of varying row number and varying col number
    if curr_row>0 and curr_row<img_rows and curr_row+row_length>img_rows and (img_rows-row_length)%params.win_shift_rows_test!=0: #Dont go to one of earlier indices
        curr_row = img_rows-row_length;
        curr_col = 0;
        while(curr_col+col_length<=img_cols):
            # newImage and predict\
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;
            curr_col+=params.win_shift_cols_test;
        # Saving a last patch in the given set of fixed rows and varying col number
        if curr_col>0 and curr_col<img_cols and curr_col+col_length>img_cols and (img_cols-col_length)%params.win_shift_cols_test!=0: #Dont go to one of earlier indices
            curr_col = img_cols-col_length;
            # make new images
            new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
            new_image = np.expand_dims(new_image, axis=0)
            pred_mask = model.predict(new_image)[0];
            # save details
            mask_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= pred_mask;
            cnt_matrix[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:1]+= 1;
            # Increment count and continue
            curr_gen_count+=1;    # Safe Check
    if np.min(cnt_matrix)==0:
        raise Exception('At least one pixel is not in any of the patches! Use np.where() to find it.')
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
    # org_names, gen_names, gen_names_ind, gen_names_ind_cnt = generate_train_data_augmentation();
    # test_list = generate_test_data();
    print('Nothing in store!!')




#def run_length_encode(mask):
#    '''
#    img: numpy array, 1 - mask, 0 - background
#    Returns run length as string formated
#    '''
#    inds = mask.flatten()
#    # print('In RLE: Dims: ',inds.shape)
#    runs = np.where(inds[1:] != inds[:-1])[0] + 2
#    runs[1::2] = runs[1::2] - runs[:-1:2]
#    rle = ' '.join([str(r) for r in runs])
#    return rle
#def run_length_enc(label, len_threshold=0):
#    from itertools import chain
#    x = label.transpose().flatten()
#    y = np.where(x > 0)[0]
#    if len(y) < 10:  # consider as empty
#        return ''
#    z = np.where(np.diff(y) > 1)[0]
#    start = np.insert(y[z+1], 0, y[0])
#    end = np.append(y[z], y[-1])
#    length = end - start
#    res = [[s+1, l+1] for s, l in zip(list(start), list(length)) if l+1>len_threshold]
#    res = list(chain.from_iterable(res))
#    return ' '.join([str(r) for r in res])

#==============================================================================
# item = r'1a11552569160f0b1ea10bedbd628ce6c14f29edec5092034c2309c556df833e';
# cv2.imshow( "[{0}x{1}x{2}]".format(image.shape[0], image.shape[1], image.shape[2]), np.hstack((image,mask)) );
# cv2.waitKey(0);
# cv2.destroyAllWindows();
# cv2.imshow( "[{0}x{1}x{2}]".format(image.shape[0], image.shape[1], image.shape[2]), np.hstack((image,r_image)) );
#==============================================================================
#==============================================================================
# # Data Augmentation(Type 1): Horizontal and Vertical patches selection
# # Data Generation(Augmentation) desired image properties
# params.win_shift_rows, params.win_shift_cols = 100, 100;
# row_length, col_length, channels = 256, 256, 3; # Desired values
# org_names = [];
# gen_names = [];
# for item in os.listdir(params.train_folder_org):
#     org_names+=[item]; print(len(org_names));
#     # Get Paths
#     image_path = os.path.join(params.train_folder_org,item+'/images');
#     mask_path = os.path.join(params.train_folder_org,item+'/masks');
#     # Load image and masks
#     image_file = os.listdir(image_path)[0];
#     image = cv2.imread(os.path.join(image_path,image_file));
#     mask_files = os.listdir(mask_path);
#     mask = [];
#     for mask_file in mask_files:
#         partial_mask = cv2.imread(os.path.join(mask_path,mask_file));
#         try:
#             mask+=partial_mask;
#         except:
#             mask = partial_mask;
#     bw_mask =  cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY); #mask[:,:,0]; bw_mask[bw_mask>0]=1;
#     # Generate
#     img_rows, img_cols, img_channels = image.shape;
#     if(img_channels<channels):
#         raise Exception('Channels<Desired Channels')
#     curr_gen_count, curr_row, curr_col = 1, 0, 0;
#     curr_row = 0;
#     while(curr_row+row_length<=img_rows):
#         curr_col = 0;
#         while(curr_col+col_length<=img_cols):
#             new_image = image[curr_row:curr_row+row_length,curr_col:curr_col+col_length,:channels];
#             new_mask = bw_mask[curr_row:curr_row+row_length,curr_col:curr_col+col_length];
#             gen_name = str(item)+'_type1'+'_'+str(curr_gen_count);
#             gen_names+=[gen_name]
#             cv2.imwrite(os.path.join(params.train_folder_gen_m1, gen_name+'.png'), new_image);
#             cv2.imwrite(os.path.join(params.train_folder_gen_m1, gen_name+'_mask.png'), new_mask);
#             curr_gen_count+=1;
#             curr_col+=params.win_shift_cols;
#         curr_row+=params.win_shift_rows;
# with open(os.path.join(params.train_folder_gen_m1,'org_names.pickle'),'wb') as opfile:
#     pk.dump(org_names, opfile); opfile.close();
# with open(os.path.join(params.train_folder_gen_m1,'gen_names.pickle'),'wb') as opfile:
#     pk.dump(gen_names, opfile); opfile.close();
# print('-'*10); print('initial: ',len(org_names)); print('final: ',int(len(gen_names))); print('-'*10);
#==============================================================================
#==============================================================================
# def rotatedRectWithMaxArea(w, h, angle):
#   """
#   Given a rectangle of size wxh that has been rotated by 'angle' (in
#   radians), computes the width and height of the largest possible
#   axis-aligned rectangle (maximal area) within the rotated rectangle.
#   """
#   if w <= 0 or h <= 0:
#     return 0,0
#   width_is_longer = w >= h
#   side_long, side_short = (w,h) if width_is_longer else (h,w)
#   # since the solutions for angle, -angle and 180-angle are all the same,
#   # if suffices to look at the first quadrant and the absolute values of sin,cos:
#   sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
#   if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
#     # half constrained case: two crop corners touch the longer side,
#     #   the other two corners are on the mid-line parallel to the longer line
#     x = 0.5*side_short
#     wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
#   else:
#     # fully constrained case: crop touches all 4 sides
#     cos_2a = cos_a*cos_a - sin_a*sin_a
#     wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
#   return wr,hr
#==============================================================================
#==============================================================================
# def rotateImage(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result
#==============================================================================
