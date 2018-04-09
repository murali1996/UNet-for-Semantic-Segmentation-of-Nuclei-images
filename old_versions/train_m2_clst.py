# -*- coding: utf-8 -*-
from __future__ import print_function #Making python 2.7 compatible
import os, numpy as np, cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import pickle
from scipy import ndimage

import params
from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from unet_upsampling2D import get_unet_big, get_unet_medium, get_unet_small

seed = params.seed;
batch_size = params.batch_size;
n_epochs = params.n_epochs;

####################################################################################################
def append_hsv_image(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV);
    image = np.concatenate((bgr_image, hsv_image), axis=-1).astype(float);
    return image;
def mean_substraction(image):
    for channel in range(image.shape[-1]):
        image[:,:,channel]-=np.mean(image[:,:,channel]);
    return image;
def prep_image_to_unet(image_path, image):
    # image = kimage.load_img(image_path); image = kimage.img_to_array(image); pimage = preprocess_input(image);
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
def batch_generator(lst, data_dir, batch_size, training_phase):
    while 1:
        np.random.shuffle(lst)
        imax = int(len(lst)/batch_size)
        for i in range(imax): # Generate data
            get_these_ids = lst[i*batch_size:(i+1)*batch_size];
            return_images = [0]*batch_size;
            return_masks = [0]*batch_size;
            for index, thisID in enumerate(get_these_ids):
                image_path = os.path.join(data_dir, thisID+'.png');
                return_images[index] = prep_image_to_unet(image_path=image_path)
                if training_phase:
                    mask_path = os.path.join(data_dir, thisID+'_mask.png');
                    return_masks[index] =  prep_mask_to_unet(mask_path);
            if training_phase:
                return_images = np.stack(return_images);
                return_masks = np.stack(return_masks);
                yield return_images, return_masks
            else:
                return_images = np.stack(return_images);
                yield return_images
def train_model(model, tr_list, val_list, data_dir, batch_size, initial_epoch, n_epochs,
                filepath1_base, filepath2_base, logs_folder, verbose=1):
    filepath1 = os.path.join(filepath1_base, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    filepath2 = os.path.join(filepath2_base, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensor_logs = TensorBoard(log_dir = logs_folder);
    adapt_LR = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=1,epsilon=1e-4);
    callbacks_list = [checkpoint1,checkpoint2,tensor_logs,adapt_LR]
    History = model.fit_generator(
            batch_generator(tr_list, data_dir, batch_size, True),
            steps_per_epoch = int(len(tr_list)/batch_size),
            epochs = n_epochs,
            verbose = verbose,
            initial_epoch = 0,
            validation_data = batch_generator(val_list, data_dir, batch_size, True),
            validation_steps = int(len(val_list)/batch_size),
            callbacks = callbacks_list)
    return model, History
####################################################################################################


if __name__=="__main__":

####### MODEL 0
####################################### LOAD TRAIN DATA #########################################################
#    folder0 = params.cluster_0_folder_aug;
#    with open(os.path.join(folder0,'gen_names.pickle'),'rb') as opfile:
#        tr_list_full = pickle.load(opfile);  opfile.close();
#    print('Length of Training Data: ',len(tr_list_full));
#    tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
####################################### TRAINING #########################################################
#    # Load Model
#    wanna_train = True # If true, start from epoch {Please check the folder and decide}
#    model_path_clst0 = params.best_model_path_m2_c0;
#    if model_path_clst0 is not None:
#        model_clst0 = load_model(model_path_clst0,
#                                 custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
#        print('UNET model initialized');
#    else:
#        model_clst0 = get_unet_medium();
#        model_clst0.load_weights(params.best_model_path_m2)
#        print('UNET model loaded');
#    # Model training
#    if wanna_train:
#        model, History = train_model(model_clst0, tr_list, val_list, folder0, batch_size, 0, n_epochs,
#                                     filepath1_base = params.base_model_weights_train_0,
#                                     filepath2_base = params.base_model_weights_validate_0,
#                                     logs_folder = params.logs_folder_0);

#==============================================================================
# ###### MODEL 1
# ###################################### LOAD TRAIN DATA #########################################################
#     folder1 = params.cluster_1_folder_aug;
#     with open(os.path.join(folder1,'gen_names.pickle'),'rb') as opfile:
#         tr_list_full = pickle.load(opfile);  opfile.close();
#     print('Length of Training Data: ',len(tr_list_full));
#     tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
# ###################################### TRAINING #########################################################
#     # Load Model
#     wanna_train = True # If true, start from epoch {Please check the folder and decide}
#     model_path_clst1 = params.best_model_path_m2_c1;
#     if model_path_clst1 is not None:
#         model_clst1 = load_model(model_path_clst1,
#                                  custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
#         print('UNET model initialized');
#     else:
#         model_clst1 = get_unet_medium();
#         model_clst1.load_weights(params.best_model_path_m2)
#         print('UNET model loaded');
#     # Model training
#     if wanna_train:
#         model, History = train_model(model_clst1, tr_list, val_list, folder1, batch_size, 0, n_epochs,
#                                      filepath1_base = params.base_model_weights_train_1,
#                                      filepath2_base = params.base_model_weights_validate_1,
#                                      logs_folder = params.logs_folder_1);
#==============================================================================

#==============================================================================
# ###### MODEL 2
# ###################################### LOAD TRAIN DATA ######################
#     folder2 = params.cluster_2_folder_aug;
#     with open(os.path.join(folder2,'gen_names.pickle'),'rb') as opfile:
#         tr_list_full = pickle.load(opfile);  opfile.close();
#     print('Length of Training Data: ',len(tr_list_full));
#     tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
# ###################################### TRAINING #############################
#     # Load Model
#     wanna_train = True # If true, start from epoch {Please check the folder and decide}
#     model_path_clst2 = params.best_model_path_m2_c2;
#     if model_path_clst2 is not None:
#         model_clst2 = load_model(model_path_clst2,
#                                  custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
#         print('UNET model initialized');
#     else:
#         model_clst2 = get_unet_medium();
#         model_clst2.load_weights(params.best_model_path_m2)
#         print('UNET model loaded');
#     # Model training
#     if wanna_train:
#         model, History = train_model(model_clst2, tr_list, val_list, folder2, batch_size, 0, n_epochs,
#                                      filepath1_base = params.base_model_weights_train_2,
#                                      filepath2_base = params.base_model_weights_validate_2,
#                                      logs_folder = params.logs_folder_2);
#==============================================================================

#==============================================================================
# ###### MODEL 3
# ###################################### LOAD TRAIN DATA ######################
#     folder3 = params.cluster_3_folder_aug;
#     with open(os.path.join(folder3,'gen_names.pickle'),'rb') as opfile:
#         tr_list_full = pickle.load(opfile);  opfile.close();
#     print('Length of Training Data: ',len(tr_list_full));
#     tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
# ###################################### TRAINING #############################
#     # Load Model
#     wanna_train = True # If true, start from epoch {Please check the folder and decide}
#     model_path_clst3 = params.best_model_path_m2_c3;
#     if model_path_clst3 is not None:
#         model_clst3 = load_model(model_path_clst3,
#                                  custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
#         print('UNET model initialized');
#     else:
#         model_clst3 = get_unet_medium();
#         model_clst3.load_weights(params.best_model_path_m2)
#         print('UNET model loaded');
#     # Model training
#     if wanna_train:
#         model, History = train_model(model_clst3, tr_list, val_list, folder3, batch_size, 0, n_epochs,
#                                      filepath1_base = params.base_model_weights_train_3,
#                                      filepath2_base = params.base_model_weights_validate_3,
#                                      logs_folder = params.logs_folder_3);
#==============================================================================



###################################### QUICK TEST ######################################################
#==============================================================================
#     model_path_clst3 = params.best_model_path_m2_c3;
#     if model_path_clst3 is not None:
#         model_clst3 = load_model(model_path_clst3,
#                                   custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
#         print('UNET model loaded from path pointed in params file');
#     items = [r'f952cc65376009cfad8249e53b9b2c0daaa3553e897096337d143c625c2df886_128x128_12_0',
#              r'ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716_128x128_9_90',
#              r'cbff60361ded0570e5d50429a1aa51d81471819bc9b38359f03cfef76de0038c_128x128_9_90',
#              r'c0152b1a260e71f9823d17f4fbb4bf7020d5dce62b4a12b3099c1c8e52a1c43a_128x128_4_0',
#              r'a7f767ca9770b160f234780e172aeb35a50830ba10dc49c526f4712451abe1d2_128x128_5_0',
#              r'4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06_128x128_6_90',
#              r'853a4c67900c411abd04467f7bc7813d3c58a5f565c8b0807e13c6e6dea21344_128x128_11_270'] #BW type
#     for item in items:
#         image_path = os.path.join(params.cluster_3_folder_aug,item+'.png');
#         mask_path = os.path.join(params.cluster_3_folder_aug,item+'_mask.png');
#         image = cv2.imread(image_path)
#         dummy_image = (image[:128,:128,:]).astype('float32')
#         prep_image = prep_image_to_unet(image_path)
#         prep_image = np.expand_dims(prep_image, axis=0)
#         pred_image = model_clst3.predict(prep_image)[0];
#         # Global Thresholding
#         temp = pred_image.copy();
#         pred_image_thres1 = 255*(temp > 0.5);
#         pred_image_thres1 = pred_image_thres1.astype(np.uint8)  # threshold
#         # Otsu's thresholding
#         temp = pred_image.copy();
#         pred_image_255 = 255*temp;
#         pred_image_255_uint8 = pred_image_255.astype(np.uint8)
#         _, pred_image_thres2 = cv2.threshold(pred_image_255_uint8, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         # Adaptive Thresholding
#         temp = pred_image.copy();
#         pred_image_255 = 255*temp;
#         pred_image_255_uint8 = pred_image_255.astype(np.uint8)
#         pred_image_thres3 = cv2.adaptiveThreshold(pred_image_255_uint8, 255,
#                                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                   thresholdType=cv2.THRESH_BINARY, blockSize=13, C=0)
#         # Global Thresholding AND Adaptive Thresholding
#         pred_image_thres4 = np.logical_or(pred_image_thres2==255,pred_image_thres3==255);
#         pred_image_thres4 = pred_image_thres4.astype(np.uint8)
#         pred_image_thres4 = 255*pred_image_thres4;
#         #((np.max(pred_image)-np.min(pred_image))/2) # threshold
#         cv2.imshow( "{0}".format(item), image);
#         cv2.imshow( "{0}_mask_true".format(item),cv2.imread(mask_path));
#         cv2.imshow( "{0}_predicted_mask_simple".format(item), pred_image_thres1);
#         cv2.imshow( "{0}_predicted_mask_otsu".format(item), pred_image_thres2);
#         cv2.imshow( "{0}_predicted_mask_adap".format(item), pred_image_thres3);
#         cv2.imshow( "{0}_predicted_mask_best".format(item), pred_image_thres4);
#         cv2.waitKey(0);
#==============================================================================


###################################### TESTING ######################################################
    #clustering_data = {'dominant_centers''n_clusters''cluster_centers_','cluster_percent_'
    #                   'image_labels''images_in_each_cluster''all_train_folders''clt'
    with open(os.path.join(params.clusters_folder_m2,'clustering_data.pickle'), 'rb') as opfile:
        clustering_data = pickle.load(opfile); opfile.close(); #n_clusters = clustering_data['n_clusters'];
    clt = clustering_data['clt'];
    # Imports required
    from data_m2_clst import make_patches_and_predict # from helpers import resize_image_dims_multiple_of
    from clustering import dominant_clusters
    from color_transfer import color_transfer
    # Deliverables
    image_paths, images, masks = [], [], [];
    #Load test data
    ts_list = load_test_list(); print('Length of Testing Data: ',len(ts_list));
    # Initialize all 4 u-net models
    models = {};
    models[0] = load_model(params.best_model_path_m2_c0, \
                          custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
    models[1] = load_model(params.best_model_path_m2_c1, \
                          custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
    models[2] = load_model(params.best_model_path_m2_c2, \
                          custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
    models[3] = load_model(params.best_model_path_m2_c3, \
                          custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
    # Run the loop =D
    for ind, item in enumerate(ts_list[5:6]):
        # Load the full image file
        print(ind);
        image_path = os.path.join(params.test_folder_gen,item+'.png')
        bgr_image = cv2.imread(os.path.join(params.test_folder_gen,item+'.png'))
        image_paths.append(image_path);
        images.append(bgr_image);
        img_rows, img_cols, img_channels = bgr_image.shape;
        # Find dominant cluster center of this image and get the cluster label
        ctr,_,_ = dominant_clusters(bgr_image, n_clusters=1);
        label = clt.predict(ctr)[0];
        print(label);
#        if label==3:
#            bgr_target = cv2.imread(params.color_transfer_target_label3);
#            bgr_image = color_transfer(bgr_target, bgr_image);
#        if label==1 or label==3:
#            bgr_target = cv2.imread(params.color_transfer_target_label1);
#            bgr_image = color_transfer(bgr_target, bgr_image);
        model = models[label];
        # Make sure the image dimensions are min required values. Else add background color of this cluster
            # image = resize_image_dims_multiple_of(256, image);
        # Make smaller patches and predict
        mask = make_patches_and_predict(model, bgr_image, print_images=False);
        # Resize the mask
            # mask = mask[:img_rows,:img_cols,0]
        masks.append(mask);
        print('Image Dims: ', img_rows, img_cols, img_channels, \
              ' Changed Dims to: ', bgr_image.shape, \
              ' Mask Dims: ', mask.shape)
        # Show Images
        cv2.imshow( "{0}".format('ThisImage'), images[-1]);
        cv2.imshow( "{0}".format('ThisTransform'), bgr_image);
        cv2.imshow( "{0}".format('ThisMask'), masks[-1]);
        cv2.waitKey(0);
    del item, image_path
    del img_rows, img_cols, img_channels
    # Show images
    cv2.imshow( "{0}".format('ThisImage'), images[-1]);
    cv2.imshow( "{0}".format('ThisMask'), masks[-1]);

#==============================================================================
#     # Erode-Dilate mask
#     for i in range(len(masks)):
#         print(i)
#         iterations1 = 1
#         iterations2 = 1
#         # kernel = np.ones((5,5),np.uint8)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         # Mask
#         mask = masks[i];
#         er_mask = cv2.erode(mask,kernel,iterations = iterations1)
#         dl_mask = cv2.dilate(er_mask,kernel, iterations = iterations2)
#         #cv2.imshow( "{0}".format('Actual Image'), images[8]);
#         cv2.imshow( "{0}{1}{2}".format('Actual Mask','Eroded_Mask','Eroded_Dilated_Mask'), \
#                     np.hstack((255*mask[:,:,0],255*er_mask,255*dl_mask)) );
#         cv2.waitKey(0);
#     del i, iterations1, iterations2, mask, kernel, dl_mask, er_mask
#==============================================================================

#==============================================================================
#     # Stats on segments (Connected Components) and delete very small segments
#     mx_segment, n_segments = [], [];
#     for i in range(len(masks)):
#         mask = masks[i];
#         label_im, nb_labels = ndimage.label(mask)
#         sizes = []
#         for j in range(nb_labels):
#             sizes+=[len(np.where(label_im==j+1)[0])]; #print(j+1, sizes[i])
#         mx_segment+=[np.max(sizes)];
#         n_segments+=[len(sizes)]
#         #np.sort(sizes)
#     del i, j, mask, label_im, nb_labels, sizes
#     # Count > threshold
#     up_thres = np.mean(sizes);
#     cnt = 0;
#     for size in sizes:
#         if size>up_thres:
#             cnt+=1
#     print(cnt)
#     del size
#==============================================================================

    # Make Final Masks and RLE of them and write to file
#==============================================================================
#     import pandas as pd
#     from helpers import rle_encoding
#     df = pd.DataFrame([], columns=['ImageId', 'EncodedPixels'])
#     ind = 0;
#     for i in range(len(masks)):
#         print(i)
#         mask = masks[i];
#         label_im, nb_labels = ndimage.label(mask)
#         for j in range(nb_labels):
#             df.loc[ind,'ImageId'] = ts_list[i];
#             df.loc[ind,'EncodedPixels'] = rle_encoding(label_im, j+1);
#             ind+=1;
#     del i, j, ind, mask, label_im, nb_labels
#     df.to_csv(os.path.join(params.submit_path,'4_4_18_sub1.csv'), index=False)
#==============================================================================



#final_mask = np.zeros(mask.shape)
#for i in range(nb_labels):
#    if len(np.where(label_im==i+1)[0])>up_thres:
#        #temp_mask = np.zeros(mask.shape)
#        #temp_mask[label_im==i+1] = 1
#        #cv2.imshow( "{0}".format(i+1), temp_mask);
#        final_mask[label_im==i+1] = 1
#del i, #temp_mask
#cv2.imshow( "{0}".format('Final_Mask'), 255*final_mask);
########################################################################################################
########################################################################################################
########################################################################################################



