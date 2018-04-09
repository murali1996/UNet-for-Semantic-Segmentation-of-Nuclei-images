# -*- coding: utf-8 -*-
from __future__ import print_function #Making python 2.7 compatible
import os, numpy as np, cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

import params
from data_m2 import load_train_list, load_test_list
from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from unet import get_unet_big, get_unet_medium, get_unet_small

seed = params.seed;
batch_size = params.batch_size;
n_epochs = params.n_epochs;

####################################################################################################
def prep_image_to_unet(image_path=None, image=None):
    # image = kimage.load_img(image_path); image = kimage.img_to_array(image); pimage = preprocess_input(image);
    if image_path is None:
        bgr_image = image.copy();
    elif image is None:
        bgr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED);
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV);
    image = np.concatenate((bgr_image, hsv_image), axis=-1).astype(float);
    for channel in range(image.shape[-1]):
        image[:,:,channel]-=np.mean(image[:,:,channel]);
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
def train_model(model, tr_list, val_list, data_dir, batch_size, n_epochs=1, weights=None, verbose=1):
    if weights is not None:
        model.load_weights(weights)
    filepath1 = os.path.join(params.base_model_weights_train, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    filepath2 = os.path.join(params.base_model_weights_validate, "weights-epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.hdf5")
    checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensor_logs = TensorBoard(log_dir = params.logs_folder)
    adapt_LR = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=1,epsilon=1e-4);
    callbacks_list = [checkpoint1,checkpoint2,tensor_logs,adapt_LR]
    History = model.fit_generator(
            batch_generator(tr_list, data_dir, batch_size, True),
            steps_per_epoch = int(len(tr_list)/batch_size),
            epochs = n_epochs,
            verbose = verbose,
            initial_epoch = 23,
            validation_data = batch_generator(val_list, data_dir, batch_size, True),
            validation_steps = int(len(val_list)/batch_size),
            callbacks = callbacks_list)
    return model, History



if __name__=="__main__":
###################################### TRAINING #########################################################
    # Load Model
    wanna_train = True # If true, start from epoch 20 {Please check the folder and decide}
    model_path = params.best_model_path_m2;
    if model_path is not 'None':
        model = load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff});
        print('UNET model initialized loaded');
    else:
        model = get_unet_medium();
        print('UNET model initialized loaded');
    # Model training
    if wanna_train:
        # Get Training List and split train-validation sets
        tr_list_full = load_train_list();
        print('Length of Training Data: ',len(tr_list_full));
        tr_list, val_list = train_test_split(tr_list_full, test_size=0.20, random_state=seed);
        model, History = train_model(model, tr_list, val_list, params.train_folder_gen_m2, batch_size, n_epochs);

###################################### TESTING ######################################################
    # from helpers import resize_image_dims_multiple_of
    from data_m2 import make_patches_and_predict
    images = [];
    masks = [];
    ts_list = load_test_list(); print('Length of Testing Data: ',len(ts_list));
#    ts_list = [r'259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
#               r'472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',] #Color type
    for ind, item in enumerate(ts_list):
        print(ind);
        image_path = os.path.join(params.test_folder_gen,item+'.png')
        bgr_image = cv2.imread(os.path.join(params.test_folder_gen,item+'.png'))
        images.append(bgr_image);
        # Preprocess
        img_rows, img_cols, img_channels = bgr_image.shape;
        # Resize and predict
        # image = resize_image_dims_multiple_of(256, image);
        mask = make_patches_and_predict(model, bgr_image);
        # mask = mask[:img_rows,:img_cols,0]
        masks.append(mask);
        print('Image Dims: ', img_rows, img_cols, img_channels, ' Changed Dims to: ',bgr_image.shape, ' Mask Dims: ', mask.shape)
        # Show images
#        cv2.imshow( "{0}".format('ThisImage'), images[-1]);
#        cv2.imshow( "{0}".format('ThisMask'), 255*masks[-1]);
#        cv2.waitKey(0)
    del item, image_path
    del img_rows, img_cols, img_channels
    # Show images
#    cv2.imshow( "{0}".format('ThisImage'), images[20]);
#    cv2.imshow( "{0}".format('ThisMask'), 255*masks[20]);

    # Erode-Dilate mask
    for i in range(len(masks)):
        print(i)
        iterations1 = 1
        iterations2 = 1
        # kernel = np.ones((5,5),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Mask
        mask = masks[i];
        er_mask = cv2.erode(mask,kernel,iterations = iterations1)
        dl_mask = cv2.dilate(er_mask,kernel, iterations = iterations2)
        #cv2.imshow( "{0}".format('Actual Image'), images[8]);
#        cv2.imshow( "{0}{1}{2}".format('Actual Mask','Eroded_Mask','Eroded_Dilated_Mask'), \
#                    np.hstack((255*mask[:,:,0],255*er_mask,255*dl_mask)) );
#        cv2.waitKey(0);
    del i, iterations1, iterations2, mask, kernel, dl_mask, er_mask

    # Stats on segments (Connected Components)
    from scipy import ndimage
    mx_segment, n_segments = [], [];
    for i in range(len(masks)):
        mask = masks[i];
        label_im, nb_labels = ndimage.label(mask)
        sizes = []
        for j in range(nb_labels):
            sizes+=[len(np.where(label_im==j+1)[0])]; #print(j+1, sizes[i])
        mx_segment+=[np.max(sizes)];
        n_segments+=[len(sizes)]
        #np.sort(sizes)
    del i, j, mask, label_im, nb_labels, sizes
#    # Count > threshold
#    up_thres = np.mean(sizes);
#    cnt = 0;
#    for size in sizes:
#        if size>up_thres:
#            cnt+=1
#    print(cnt)
#    del size

    # Make Final Masks and RLE of them and write to file
    import pandas as pd
    from helpers import rle_encoding
    df = pd.DataFrame([], columns=['ImageId', 'EncodedPixels'])
    ind = 0;
    for i in range(len(masks)):
        print(i)
        mask = masks[i][:,:,0];
        label_im, nb_labels = ndimage.label(mask)
        for j in range(nb_labels):
            df.loc[ind,'ImageId'] = ts_list[i];
            df.loc[ind,'EncodedPixels'] = rle_encoding(label_im, j+1);
            ind+=1;
    del i, j, ind, mask, label_im, nb_labels
    df.to_csv(os.path.join(params.submit_path,'29_3_18_sub2.csv'), index=False)



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

###################################### QUICK TEST ######################################################
    # Sample check test image results
#    items = [r'0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5',
#             r'8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab',
#             r'53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4',
#             r'550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18'] #BW type
#    for item in items:
#        image_path = os.path.join(params.test_folder_gen,item+'.png');
#        dummy_image1 = cv2.imread(image_path)
#        dummy_image = (dummy_image1[:128,:128,:]).astype('float32')
#        prep_image = prep_image_to_unet(image_path)
#        prep_image = np.expand_dims(prep_image, axis=0)
#        pred_image = model.predict(prep_image)
#        pred_image = pred_image[0];
#        pred_image = 255*(pred_image > 0.5).astype(np.uint8)  # threshold
#        #((np.max(pred_image)-np.min(pred_image))/2) # threshold
#        cv2.imshow( "{0}".format(item), dummy_image1);
#        cv2.imshow( "{0}_mask".format(item), pred_image);
#    items = [r'259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4',
#             r'472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71',] #Color type
#    for item in items:
#        dummy_image1 = cv2.imread(os.path.join(params.test_folder_gen,item+'.png'))
#        dummy_image = (dummy_image1[:256,:256,:]).astype('float32')
#        prep_image = preprocess_input(dummy_image)
#        prep_image = np.expand_dims(prep_image, axis=0)
#        pred_image = model.predict(prep_image)
#        pred_image = pred_image[0];
#        # pred_image = 255*(pred_image > 0.5).astype(np.uint8)  # threshold
#        pred_image = 255*(pred_image > ((np.max(pred_image)-np.min(pred_image))/2)).astype(np.uint8)  # threshold
#        cv2.imshow( "{0}".format(item), dummy_image);
#        cv2.imshow( "{0}_mask".format(item), pred_image);
#    del items, item, dummy_image1, dummy_image, prep_image, pred_image

#==============================================================================
# def test_model(model,ts_data,ts_labels,weights):
#     if weights is not None:
#         model.load_weights(weights)
#     scores = model.evaluate(ts_data,ts_labels, verbose=1)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     return model, scores
#==============================================================================
#==============================================================================
# # Closing-OPening
# import cv2
# img = images[1];
# img_bw = (255*masks[1]).astype('uint8').copy();
# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
# mask = np.dstack([mask, mask, mask]) / 255
# out = img * mask
# cv2.imshow('Output', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('output.png', out)
# del img, img_bw, se1, se2, mask, out
#==============================================================================
