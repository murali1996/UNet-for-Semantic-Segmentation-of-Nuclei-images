# -*- coding: utf-8 -*-
from __future__ import print_function #Making python 2.7 compatible
import os, numpy as np, cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing import image as kimage
from keras.models import load_model
#from keras.applications.vgg16 import preprocess_input
import params
from data_m3 import load_train_list_with_augmentation, load_test_list
from losses import bce_dice_loss, dice_coeff
from unet_transConv2D import get_unet_medium

seed = params.seed;
batch_size = params.batch_size;
n_epochs = params.n_epochs;

####################################################################################################
def prep_image_to_unet(image_path, image=None):
    if image_path is not None:
        bgr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED);
    else:
        bgr_image = image.copy();
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB);
    bgr_image = bgr_image.astype(float)
    ycrcb_image = ycrcb_image.astype(float)
    #image = np.concatenate((bgr_image, hsv_image), axis=-1).astype(float);
    for channel in range(bgr_image.shape[-1]):
        bgr_image[:,:,channel]-=np.mean(bgr_image[:,:,channel]);
    for channel in range(ycrcb_image.shape[-1]):
        ycrcb_image[:,:,channel]-=np.mean(ycrcb_image[:,:,channel]);
    return bgr_image, ycrcb_image
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
            return_images_bgr = [0]*batch_size;
            return_images_ycrcb = [0]*batch_size;
            return_masks = [0]*batch_size;
            for index, thisID in enumerate(get_these_ids):
                image_path = os.path.join(data_dir, thisID+'.png');
                return_images_bgr[index], return_images_ycrcb[index] = prep_image_to_unet(image_path);
                if training_phase:
                    mask_path = os.path.join(data_dir, thisID+'_mask.png');
                    return_masks[index] =  prep_mask_to_unet(mask_path);
            if training_phase:
                return_images_bgr = np.stack(return_images_bgr);
                return_images_ycrcb = np.stack(return_images_ycrcb);
                return_masks = np.stack(return_masks);
                yield [return_images_ycrcb,return_images_bgr], return_masks
            else:
                return_images_bgr = np.stack(return_images_bgr);
                return_images_ycrcb = np.stack(return_images_ycrcb);
                yield [return_images_ycrcb,return_images_bgr]
def train_model(model, tr_list, val_list, data_dir, batch_size, n_epochs, verbose=1):
    filepath1 = os.path.join(params.base_model_weights_train, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    filepath2 = os.path.join(params.base_model_weights_validate, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensor_logs = TensorBoard(log_dir = params.logs_folder)
    #adapt_LR = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=1,epsilon=1e-4);
    adapt_LR = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=1,epsilon=1e-4);
    callbacks_list = [checkpoint1, checkpoint2, tensor_logs, adapt_LR]
    History = model.fit_generator(
            batch_generator(tr_list, data_dir, batch_size, True),
            steps_per_epoch = int(len(tr_list)/batch_size),
            epochs = n_epochs,
            verbose = verbose,
            initial_epoch = 37,
            validation_data = batch_generator(val_list, data_dir, batch_size, True),
            validation_steps = int(len(val_list)/batch_size),
            callbacks = callbacks_list)
    return model, History



if __name__=="__main__":
###################################### TRAINING #########################################################
    # Load Model
    wanna_train = False # If true, start from epoch 30 {Please check the folder and decide}
    model_path = params.best_model_path_m3;
    if model_path is not None:
        model = load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
        print('UNET model initialized loaded');
    else:
        model = get_unet_medium();
        print('UNET model initialized loaded');
    # Model training
    if wanna_train:
        target_folder = params.train_folder_gen_m3
        # Get Training List and split train-validation sets
        tr_list_full = load_train_list_with_augmentation(target_folder);
        print('Length of Training Data: ',len(tr_list_full));
        tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
        #model, History = train_model(model, tr_list, val_list, params.train_folder_gen_m3, batch_size, n_epochs);
        model, History = train_model(model, tr_list, val_list, target_folder, batch_size, n_epochs);
####################################### Quick TESTING BY RESIZING ######################################################
#    items = [r'0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5',
#             r'8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab',
#             r'53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4',
#             r'550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18'] #BW type
#    items = [r'259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
#             r'472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',] #Color type
#    for item in items:
#        image_path = os.path.join(params.test_folder_gen,item+'.png');
#        image = cv2.imread(image_path);
#        resized_image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR);
#        #resized_image = image[:256,:256,:].copy();
#        bgr_image, ycrcb_image = prep_image_to_unet(None, resized_image)
#        bgr_image = np.expand_dims(bgr_image, axis=0)
#        ycrcb_image = np.expand_dims(ycrcb_image, axis=0)
#        pred_image = model.predict([ycrcb_image, bgr_image])[0];
#        pred_image = 255*(pred_image > 0.5) # threshold
#        pred_image = pred_image.astype(np.uint8);
#        pred_image = cv2.resize(pred_image, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR);
#        cv2.imshow( "{0}".format('image'), image);
#        #cv2.imshow( "{0}".format('resized_image'), resized_image);
#        cv2.imshow( "{0}_mask".format('image'), pred_image);
#        cv2.waitKey(0);
####################################### FULL TESTING BY RESIZING ######################################################
#    images = [];
#    masks = [];
#    test_images_folder = params.test2_folder_gen;
#    ts_list = load_test_list(test_images_folder);
#    print('Length of Testing Data: ',len(ts_list));
#    for ind, item in enumerate(ts_list):
#        print(ind);
#        image_path = os.path.join(test_images_folder,item+'.png');
#        image = cv2.imread(image_path);
#        images.append(image);
#        resized_image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR);
#        bgr_image, ycrcb_image = prep_image_to_unet(None, resized_image)
#        bgr_image = np.expand_dims(bgr_image, axis=0)
#        ycrcb_image = np.expand_dims(ycrcb_image, axis=0)
#        pred_image = model.predict([ycrcb_image, bgr_image])[0];
#        pred_image[pred_image > 0.5] = 1; # threshold
#        pred_image = pred_image.astype(np.uint8);
#        ##resize_pred_image_lin = cv2.resize(pred_image, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR);
#        resize_pred_image_cub = cv2.resize(pred_image, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_CUBIC);
#        masks.append(resize_pred_image_cub);
#        #cv2.imshow( "{0}".format('image'), image);
#        ##cv2.imshow( "{0}_mask_linear".format('image'), 255*resize_pred_image_lin);
#        #cv2.imshow( "{0}_mask_cubic".format('image'), 255*resize_pred_image_cub);
#        #cv2.waitKey(0);
#        cv2.imwrite(os.path.join(test_images_folder,item+'_mask.png'),255*resize_pred_image_cub);
####################################### TESTING  BY PATCHES ######################################################
#    from data_m3 import make_patches_and_predict
#    images = [];
#    masks = [];
#    ts_list = load_test_list(); print('Length of Testing Data: ',len(ts_list));
#    for ind, item in enumerate(ts_list[0:1]):
#        print(ind);
#        image_path = os.path.join(params.test_folder_gen,item+'.png')
#        bgr_image = cv2.imread(os.path.join(params.test_folder_gen,item+'.png'))
#        images.append(bgr_image);
#        # Preprocess
#        img_rows, img_cols, img_channels = bgr_image.shape;
#        # Resize and predict
#        # image = resize_image_dims_multiple_of(256, image);
#        mask = make_patches_and_predict(model, bgr_image);
#        # mask = mask[:img_rows,:img_cols,0]
#        masks.append(mask);
#        print('Image Dims: ', img_rows, img_cols, img_channels, ' Changed Dims to: ',bgr_image.shape, ' Mask Dims: ', mask.shape)
#        # Show images
#        cv2.imshow( "{0}".format('ThisImage'), images[-1]);
#        cv2.imshow( "{0}".format('ThisMask'), 255*masks[-1]);
#        cv2.waitKey(0)
#    del item, image_path
#    del img_rows, img_cols, img_channels
####################################### POST-PROCESSING ######################################################
    #Show images
#    cv2.imshow( "{0}".format('ThisImage'), images[20]);
#    cv2.imshow( "{0}".format('ThisMask'), 255*masks[20]);
    # Erode-Dilate mask
#    for i in range(len(masks)):
#        print(i)
#        iterations1 = 1
#        iterations2 = 1
#        # kernel = np.ones((5,5),np.uint8)
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#        # Mask
#        mask = masks[i];
#        er_mask = cv2.erode(mask,kernel,iterations = iterations1)
#        dl_mask = cv2.dilate(er_mask,kernel, iterations = iterations2)
#        #cv2.imshow( "{0}".format('Actual Image'), images[i]);
#        #cv2.imshow( "{0}{1}{2}".format('Actual Mask','Eroded_Mask','Eroded_Dilated_Mask'), \
#        #            np.hstack((255*mask,255*er_mask,255*dl_mask)) );
#        #cv2.waitKey(0);
#    del i, iterations1, iterations2, mask, kernel, dl_mask, er_mask
####################################### EXTRACTING SEGMENTS ##################################################
    # Stats on segments (Connected Components)
#    from scipy import ndimage
#    mx_segment, n_segments = [], [];
#    for i in range(len(masks)):
#        mask = masks[i];
#        label_im, nb_labels = ndimage.label(mask)
#        sizes = []
#        for j in range(nb_labels):
#            sizes+=[len(np.where(label_im==j+1)[0])]; #print(j+1, sizes[i])
#        mx_segment+=[np.max(sizes)];
#        n_segments+=[len(sizes)]
#        #np.sort(sizes)
#    del i, j, mask, label_im, nb_labels, sizes
#    #Count > threshold
#    up_thres = np.mean(sizes);
#    cnt = 0;
#    for size in sizes:
#        if size>up_thres:
#            cnt+=1
#    print(cnt)
#    del size

####################################### RLE ######################################################
#    import pandas as pd
#    from helpers import rle_encoding
#    from scipy import ndimage
#    df = pd.DataFrame([], columns=['ImageId', 'EncodedPixels'])
#    ind = 0;
#    for i in range(len(masks)):
#        print(i)
#        mask = masks[i];
#        label_im, nb_labels = ndimage.label(mask)
#        for j in range(nb_labels):
#            df.loc[ind,'ImageId'] = ts_list[i];
#            df.loc[ind,'EncodedPixels'] = rle_encoding(label_im, j+1);
#            ind+=1;
#    del i, j, ind, mask, label_im, nb_labels
#    df.to_csv(os.path.join(params.submit_path,'13_4_18_sub1_test2.csv'), index=False)
#########################################################################################################
#########################################################################################################
#########################################################################################################
