# vein_finder_segmentation

input_1 (InputLayer)            (None, 256, 256, 1)     **DONE**   

conv2d_1 (Conv2D)               (None, 256, 256, 16)    **DONE**    

conv2d_2 (Conv2D)               (None, 256, 256, 16)    **DONE**

max_pooling2d_1 (MaxPooling2D)  (None, 128, 128, 16)    **DONE**

conv2d_3 (Conv2D)               (None, 128, 128, 32)    **DONE**

conv2d_4 (Conv2D)               (None, 128, 128, 32)    **DONE**

max_pooling2d_2 (MaxPooling2D)  (None, 64, 64, 32)      **DONE**

conv2d_5 (Conv2D)               (None, 64, 64, 64)      **DONE**

conv2d_6 (Conv2D)               (None, 64, 64, 64)      **DONE**

max_pooling2d_3 (MaxPooling2D)  (None, 32, 32, 64)

conv2d_7 (Conv2D)               (None, 32, 32, 64)

conv2d_8 (Conv2D)               (None, 32, 32, 64)

dropout_1 (Dropout)             (None, 32, 32, 64)

max_pooling2d_4 (MaxPooling2D)  (None, 16, 16, 64) 

conv2d_9 (Conv2D)               (None, 16, 16, 128)

conv2d_10 (Conv2D)              (None, 16, 16, 128)

dropout_2 (Dropout)             (None, 16, 16, 128)

up_sampling2d_1 (UpSampling2D)  (None, 32, 32, 128) 

conv2d_11 (Conv2D)              (None, 32, 32, 64)

concatenate_1 (Concatenate)     (None, 32, 32, 128)

conv2d_12 (Conv2D)              (None, 32, 32, 64)

conv2d_13 (Conv2D)              (None, 32, 32, 64)

up_sampling2d_2 (UpSampling2D)  (None, 64, 64, 64)

conv2d_14 (Conv2D)              (None, 64, 64, 32)

concatenate_2 (Concatenate)     (None, 64, 64, 96)

conv2d_15 (Conv2D)              (None, 64, 64, 64)

conv2d_16 (Conv2D)              (None, 64, 64, 64)

up_sampling2d_3 (UpSampling2D)  (None, 128, 128, 64)

conv2d_17 (Conv2D)              (None, 128, 128, 32)

concatenate_3 (Concatenate)     (None, 128, 128, 64)

conv2d_18 (Conv2D)              (None, 128, 128, 32)

conv2d_19 (Conv2D)              (None, 128, 128, 32)

up_sampling2d_4 (UpSampling2D)  (None, 256, 256, 32)

conv2d_20 (Conv2D)              (None, 256, 256, 16)

concatenate_4 (Concatenate)     (None, 256, 256, 32) 

conv2d_21 (Conv2D)              (None, 256, 256, 16)

conv2d_22 (Conv2D)              (None, 256, 256, 16)

conv2d_23 (Conv2D)              (None, 256, 256, 2)

conv2d_24 (Conv2D)              (None, 256, 256, 1) 
