**using cudnn performance**

| Operation | without using cudnn (miliSecond) | with using cudnn (milimSecond) |
| :---         |     :---:      |          ---: |
| conv2d_2  | 0.063181          |0.028133     |
| conv2d_3     | 1.367092       |    0.021935       |
| conv2d_4     | 2.789974       |    0.027895       |
| conv2d_5     | 1.690865       | 0.030994       |
| conv2d_6     | 3.129005       | 0.041962      |
| conv2d_7     | 3.093958       | 0.015020       |
| conv2d_8     | 3.164053       | 0.012159      |
| conv2d_9     | 4.925966        | 0.015974       |
| conv2d_10     | 9.057999        | 0.015020       |
| conv2d_11     | 6.165028       | 0.014067      |
| conv2d_12     | 6.513119        | 0.014067      |
| conv2d_13     |3.114939      | 0.041008      |
| conv2d_14     | 3.297091      | 0.014067      |
| conv2d_15     | 6.387949       | 0.013828      |
| conv2d_16     | 3.302097      | 0.015020      |
| conv2d_17     | 4.734039       | 0.014782      |
| conv2d_18     | 5.023003        | 0.014067      |
| conv2d_19    |2.573013       | 0.017166     |
| conv2d_20     | 4.115105      | 0.018120      |
| conv2d_21     | 4.421949       | 0.017166      |
| conv2d_22     | 3.973961      | 0.015974      |
| conv2d_23     | 0.527859       | 0.016212      |


| Performance | without using cudnn (miliSecond) | with using cudnn (milimSecond) |
| :---         |     :---:      |          ---: |
|     | 0.089542 sec (11 fps)      | 0.002996 sec  (333.71 fps)      |

# Network

| Network Layer | Shape | Status |
| :---         |     :---:      |          ---: |
| input_1 (InputLayer)     |       (None, 256, 256, 1)   |  **DONE**|   
| conv2d_1 (Conv2D)        |       (None, 256, 256, 16)  |  **DONE** |   
| conv2d_2 (Conv2D)        |       (None, 256, 256, 16)  |  **DONE** |
| max_pooling2d_1 (MaxPooling2D)  | (None, 128, 128, 16) |    **DONE**| 
| conv2d_3 (Conv2D)            |   (None, 128, 128, 32) |    **DONE**| 
| conv2d_4 (Conv2D)              |  (None, 128, 128, 32)  |   **DONE**| 
| max_pooling2d_2 (MaxPooling2D) |  (None, 64, 64, 32)   |    **DONE**| 
| conv2d_5 (Conv2D)             |   (None, 64, 64, 64)   |    **DONE**| 
| conv2d_6 (Conv2D)             |   (None, 64, 64, 64)   |    **DONE**| 
| max_pooling2d_3 (MaxPooling2D) |  (None, 32, 32, 64)   |    **DONE**| 
| conv2d_7 (Conv2D)             |   (None, 32, 32, 64)   |    **DONE**| 
| conv2d_8 (Conv2D)              |  (None, 32, 32, 64)   |    **DONE**| 
| dropout_1 (Dropout)            |  (None, 32, 32, 64)   |    **DONE**| 
| max_pooling2d_4 (MaxPooling2D)|   (None, 16, 16, 64)   |    **DONE**| 
| conv2d_9 (Conv2D)             |   (None, 16, 16, 128)  |    **DONE**| 
| conv2d_10 (Conv2D)            |   (None, 16, 16, 128)  |    **DONE**| 
| dropout_2 (Dropout)            |  (None, 16, 16, 128)  |    **DONE**| 
| up_sampling2d_1 (UpSampling2D) |  (None, 32, 32, 128)  |    **DONE**| 
| conv2d_11 (Conv2D)             |  (None, 32, 32, 64)  |     **DONE**| 
| concatenate_1 (Concatenate)   |   (None, 32, 32, 128)  |    **DONE**| 
| conv2d_12 (Conv2D)             | (None, 32, 32, 64)   |    **DONE**| 
| conv2d_13 (Conv2D)            |   (None, 32, 32, 64)  |     **DONE**| 
| up_sampling2d_2 (UpSampling2D) |  (None, 64, 64, 64)  |     **DONE**| 
| conv2d_14 (Conv2D)             |  (None, 64, 64, 32) |      **DONE**| 
| concatenate_2 (Concatenate)    |  (None, 64, 64, 96)  |     **DONE**| 
| conv2d_15 (Conv2D)            |  (None, 64, 64, 64)   |   **DONE**| 
| conv2d_16 (Conv2D)              | (None, 64, 64, 64)  |     **DONE**| 
| up_sampling2d_3 (UpSampling2D) |  (None, 128, 128, 64)  |   **DONE**| 
| conv2d_17 (Conv2D)            |   (None, 128, 128, 32)  |   **DONE**| 
| concatenate_3 (Concatenate)   |   (None, 128, 128, 64)   |  **DONE**| 
| conv2d_18 (Conv2D)            |   (None, 128, 128, 32)|    **DONE**| 
| conv2d_19 (Conv2D)            |   (None, 128, 128, 32)  |   **DONE**| 
| up_sampling2d_4 (UpSampling2D) |  (None, 256, 256, 32)  |   **DONE**| 
| conv2d_20 (Conv2D)            |   (None, 256, 256, 16)  |   **DONE**| 
| concatenate_4 (Concatenate)   |   (None, 256, 256, 32)  |   **DONE**| 
| conv2d_21 (Conv2D)            |   (None, 256, 256, 16)|    **DONE**| 
| conv2d_22 (Conv2D)          |     (None, 256, 256, 16)  |   **DONE**| 
| conv2d_23 (Conv2D)          |     (None, 256, 256, 2)   |   **DONE**| 
| conv2d_24 (Conv2D)          |     (None, 256, 256, 1)   |   **DONE**| 

