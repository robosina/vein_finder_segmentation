#ifndef LAYER_H
#define LAYER_H

struct layer;
typedef struct layer layer;

struct layer{
    int width;    //kernel width
    int height;   //kernel height
    int depth;    //kernel depth (volume)
    float* weight;
    int nfilters;   //number of filters
    float* output;
    float* bias;

    int input_size;
    int output_size;
    int kernel_size;
    int bias_size;

    int im_w;
    int im_h;
};
#endif // LAYER_H

