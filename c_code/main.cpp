#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <QFile>
#include <QTextStream>
#include <iostream>
#include <layer.h>
#include <vector>
#include <thread>
#include <QImage>
#define read_from_c1 1
QString exe_path;
using namespace std;
enum LAYER
{
    CONV2D_1,
    CONV2D_2,
    MAXP2D_1,
    CONV2D_3,
    CONV2D_4,
    MAXP2D_2,
    CONV2D_5,
    CONV2D_6,
    MAXP2D_3,
    CONV2D_7,
    CONV2D_8,
    MAXP2D_4,
    CONV2D_9,
    CONV2D_10,
    UPSM2D_1,
    CONV2D_11,
    CONCAT_1,
    CONV2D_12,
    CONV2D_13,
    UPSM2D_2,
    CONV2D_14,
    CONCAT_2,
    CONV2D_15,
    CONV2D_16,
    UPSM2D_3,
    CONV2D_17,
    CONCAT_3,
    CONV2D_18,
    CONV2D_19,
    UPSM2D_4,
    CONV2D_20,
    CONCAT_4,
    CONV2D_21,
    CONV2D_22,
    CONV2D_23,
    CONV2D_24
};
vector<float> weights_conv2d_1;   //first layer
vector<float> bias_conv2d_1;      //first layer
vector<float> weights_conv2d_2;   //second layer
vector<float> bias_conv2d_2;      //second layer
vector<float> weights_conv2d_3;   //4th layer
vector<float> bias_conv2d_3;      //4th layer
vector<float> weights_conv2d_4;   //5th layer
vector<float> bias_conv2d_4;      //5th layer
vector<float> weights_conv2d_5;   //7th layer
vector<float> bias_conv2d_5;      //7th layer
vector<float> weights_conv2d_6;   //8th layer
vector<float> bias_conv2d_6;      //8th layer
vector<float> weights_conv2d_7;   //10th layer
vector<float> bias_conv2d_7;      //10th layer
vector<float> weights_conv2d_8;   //11th layer
vector<float> bias_conv2d_8;      //11th layer
vector<float> weights_conv2d_9;   //14th layer
vector<float> bias_conv2d_9;      //14th layer
vector<float> weights_conv2d_10;   //15th layer
vector<float> bias_conv2d_10;      //15th layer
vector<float> weights_conv2d_11;   //18th layer
vector<float> bias_conv2d_11;      //18th layer
vector<float> weights_conv2d_12;   //20th layer
vector<float> bias_conv2d_12;      //20th layer
vector<float> weights_conv2d_13;   //21th layer
vector<float> bias_conv2d_13;      //21th layer
vector<float> weights_conv2d_14;   //23th layer
vector<float> bias_conv2d_14;      //23th layer
vector<float> weights_conv2d_15;   //25th layer
vector<float> bias_conv2d_15;      //25th layer
vector<float> weights_conv2d_16;   //26th layer
vector<float> bias_conv2d_16;      //26th layer
vector<float> weights_conv2d_17;   //28th layer
vector<float> bias_conv2d_17;      //28th layer
vector<float> weights_conv2d_18;   //28th layer
vector<float> bias_conv2d_18;      //28th layer
vector<float> weights_conv2d_19;   //30th layer
vector<float> bias_conv2d_19;      //30th layer
vector<float> weights_conv2d_20;   //32th layer
vector<float> bias_conv2d_20;      //32th layer
vector<float> weights_conv2d_21;   //35th layer
vector<float> bias_conv2d_21;      //35th layer
vector<float> weights_conv2d_22;   //36th layer
vector<float> bias_conv2d_22;      //36th layer
vector<float> weights_conv2d_23;   //37th layer
vector<float> bias_conv2d_23;      //37th layer
vector<float> weights_conv2d_24;   //38th layer
vector<float> bias_conv2d_24;      //38th layer

extern "C" void conv2d_1(float* img_ptr,int w,int h,layer l);
extern "C" void conv2d_2(int w, int h, layer l);
extern "C" void maxp2d_1(int w,int h,layer l);
extern "C" void conv2d_3(int w,int h,layer l);
extern "C" void conv2d_4(int w,int h,layer l);
extern "C" void maxp2d_2(int w, int h, layer l);
extern "C" void conv2d_5(int w, int h, layer l);
extern "C" void conv2d_6(int w, int h, layer l);
extern "C" void maxp2d_3(int w, int h, layer l);
extern "C" void conv2d_7(int w, int h, layer l);
extern "C" void conv2d_8(int w, int h, layer l);
extern "C" void maxp2d_4(int w, int h, layer l);
extern "C" void conv2d_9(int w, int h, layer l);
extern "C" void conv2d_10(int w, int h, layer l);
extern "C" void upsample_2d_1(int w, int h, layer l);
extern "C" void conv2d_11(int w, int h, layer l);
extern "C" void concat_1(int w, int h, layer l);
extern "C" void conv2d_12(int w, int h, layer l);
extern "C" void conv2d_13(int w, int h, layer l);
extern "C" void upsample_2d_2( int w, int h, layer l);
extern "C" void conv2d_14( int w, int h, layer l);
extern "C" void concat_2( int w, int h, layer l);
extern "C" void conv2d_15(int w, int h, layer l);
extern "C" void conv2d_16(int w, int h, layer l);
extern "C" void upsample_2d_3( int w, int h, layer l);
extern "C" void conv2d_17( int w, int h, layer l);
extern "C" void concat_3(int w, int h, layer l);
extern "C" void conv2d_18(int w, int h, layer l);
extern "C" void conv2d_19(int w, int h, layer l);
extern "C" void upsample_2d_4(int w, int h, layer l);
extern "C" void conv2d_20(int w, int h, layer l);
extern "C" void concat_4(int w, int h, layer l);
extern "C" void conv2d_21(int w, int h, layer l);
extern "C" void conv2d_22(int w, int h, layer l);
extern "C" void conv2d_23(int w, int h, layer l);
extern "C" void conv2d_24(float** output, int w, int h, layer l);


extern "C" void LOAD_NEURAL_NETWORK(LAYER Layer, int w, int h, layer& l);
extern "C" void Remove_NN();
void readLines(QString filename,vector<float>& in_vec)
{
    QFile weight_file(filename);
    weight_file.open(QIODevice::ReadOnly);
    QTextStream in(&weight_file);
    while (!in.atEnd())
    {
        QString readLine=in.readLine();
        QStringList lis=readLine.split(",");
        for (int i = 0; i < lis.size(); ++i)
        {
            in_vec.push_back(lis[i].toFloat());
        }
    }
}

vector<float> read_file(QString filename)
{
    vector<float> lineF;
    readLines(filename,lineF);
    return lineF;
}
layer initialize_conv2d_1_layer()
{
    layer l;
    l.depth=1;
    l.width=3;
    l.height=3;
    l.nfilters=16;
    QString fileName=exe_path+"conv2d_1_weights.txt";
    QString bias_file=exe_path+"conv2d_1_bias.txt";
    weights_conv2d_1=read_file(fileName);
    bias_conv2d_1=read_file(bias_file);
    l.bias=&bias_conv2d_1[0];
    l.weight=&weights_conv2d_1[0];
    return l;
}

layer initialize_conv2d_2_layer()
{
    layer l;
    l.depth=16;
    l.width=3;
    l.height=3;
    l.nfilters=16;
    QString fileName=exe_path+"conv2d_2_weights.txt";
    QString bias_file=exe_path+"conv2d_2_bias.txt";
    weights_conv2d_2=read_file(fileName);
    bias_conv2d_2=read_file(bias_file);
    l.bias=&bias_conv2d_2[0];
    l.weight=&weights_conv2d_2[0];
    return l;
}

layer initialize_max_pooling2d_1()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=16;
    return l;
}

layer initialize_conv2d_3_layer()
{
    layer l;
    l.depth=16;
    l.width=3;
    l.height=3;
    l.nfilters=32;
    QString fileName=exe_path+"conv2d_3_weights.txt";
    QString bias_file=exe_path+"conv2d_3_bias.txt";
    weights_conv2d_3=read_file(fileName);
    bias_conv2d_3=read_file(bias_file);
    l.bias=&bias_conv2d_3[0];
    l.weight=&weights_conv2d_3[0];
    return l;
}

layer initialize_conv2d_4_layer()
{
    layer l;
    l.depth=32;
    l.width=3;
    l.height=3;
    l.nfilters=32;
    QString fileName=exe_path+"conv2d_4_weights.txt";
    QString bias_file=exe_path+"conv2d_4_bias.txt";
    weights_conv2d_4=read_file(fileName);
    bias_conv2d_4=read_file(bias_file);
    l.bias=&bias_conv2d_4[0];
    l.weight=&weights_conv2d_4[0];
    return l;
}

layer initialize_max_pooling2d_2()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=32;
    return l;
}

layer initialize_conv2d_5_layer()
{
    layer l;
    l.depth=32;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_5_weights.txt";
    QString bias_file=exe_path+"conv2d_5_bias.txt";
    weights_conv2d_5=read_file(fileName);
    bias_conv2d_5=read_file(bias_file);
    l.bias=&bias_conv2d_5[0];
    l.weight=&weights_conv2d_5[0];
    return l;
}

layer initialize_conv2d_6_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_6_weights.txt";
    QString bias_file=exe_path+"conv2d_6_bias.txt";
    weights_conv2d_6=read_file(fileName);
    bias_conv2d_6=read_file(bias_file);
    l.bias=&bias_conv2d_6[0];
    l.weight=&weights_conv2d_6[0];
    return l;
}

layer initialize_max_pooling2d_3()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=64;
    return l;
}

layer initialize_conv2d_7_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_7_weights.txt";
    QString bias_file=exe_path+"conv2d_7_bias.txt";
    weights_conv2d_7=read_file(fileName);
    bias_conv2d_7=read_file(bias_file);
    l.bias=&bias_conv2d_7[0];
    l.weight=&weights_conv2d_7[0];
    return l;
}

layer initialize_conv2d_8_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_8_weights.txt";
    QString bias_file=exe_path+"conv2d_8_bias.txt";
    weights_conv2d_8=read_file(fileName);
    bias_conv2d_8=read_file(bias_file);
    l.bias=&bias_conv2d_8[0];
    l.weight=&weights_conv2d_8[0];
    return l;
}

layer initialize_max_pooling2d_4()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=64;
    return l;
}

layer initialize_conv2d_9_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=128;
    QString fileName=exe_path+"conv2d_9_weights.txt";
    QString bias_file=exe_path+"conv2d_9_bias.txt";
    weights_conv2d_9=read_file(fileName);
    bias_conv2d_9=read_file(bias_file);
    l.bias=&bias_conv2d_9[0];
    l.weight=&weights_conv2d_9[0];
    return l;
}

layer initialize_conv2d_10_layer()
{
    layer l;
    l.depth=128;
    l.width=3;
    l.height=3;
    l.nfilters=128;
    QString fileName=exe_path+"conv2d_10_weights.txt";
    QString bias_file=exe_path+"conv2d_10_bias.txt";
    weights_conv2d_10=read_file(fileName);
    bias_conv2d_10=read_file(bias_file);
    l.bias=&bias_conv2d_10[0];
    l.weight=&weights_conv2d_10[0];
    return l;
}
layer initialize_up_sampling2d_1()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=128;
    return l;
}
layer initialize_conv2d_11_layer()
{
    layer l;
    l.depth=128;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_11_weights.txt";
    QString bias_file=exe_path+"conv2d_11_bias.txt";
    weights_conv2d_11=read_file(fileName);
    bias_conv2d_11=read_file(bias_file);
    l.bias=&bias_conv2d_11[0];
    l.weight=&weights_conv2d_11[0];
    return l;
}
layer initialize_concat_1()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=128;
    return l;
}
layer initialize_conv2d_12_layer()
{
    layer l;
    l.depth=128;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_12_weights.txt";
    QString bias_file=exe_path+"conv2d_12_bias.txt";
    weights_conv2d_12=read_file(fileName);
    bias_conv2d_12=read_file(bias_file);
    l.bias=&bias_conv2d_12[0];
    l.weight=&weights_conv2d_12[0];
    return l;
}
layer initialize_conv2d_13_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_13_weights.txt";
    QString bias_file=exe_path+"conv2d_13_bias.txt";
    weights_conv2d_13=read_file(fileName);
    bias_conv2d_13=read_file(bias_file);
    l.bias=&bias_conv2d_13[0];
    l.weight=&weights_conv2d_13[0];
    return l;
}

layer initialize_up_sampling2d_2()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=64;
    return l;
}
layer initialize_conv2d_14_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_14_weights.txt";
    QString bias_file=exe_path+"conv2d_14_bias.txt";
    weights_conv2d_14=read_file(fileName);
    bias_conv2d_14=read_file(bias_file);
    l.bias=&bias_conv2d_14[0];
    l.weight=&weights_conv2d_14[0];
    return l;
}

layer initialize_concat_2()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=128;
    return l;
}

layer initialize_conv2d_15_layer()
{
    layer l;
    l.depth=128;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_15_weights.txt";
    QString bias_file=exe_path+"conv2d_15_bias.txt";
    weights_conv2d_15=read_file(fileName);
    bias_conv2d_15=read_file(bias_file);
    l.bias=&bias_conv2d_15[0];
    l.weight=&weights_conv2d_15[0];
    return l;
}

layer initialize_conv2d_16_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=64;
    QString fileName=exe_path+"conv2d_16_weights.txt";
    QString bias_file=exe_path+"conv2d_16_bias.txt";
    weights_conv2d_16=read_file(fileName);
    bias_conv2d_16=read_file(bias_file);
    l.bias=&bias_conv2d_16[0];
    l.weight=&weights_conv2d_16[0];
    return l;
}
layer initialize_up_sampling2d_3()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=64;
    return l;
}
layer initialize_conv2d_17_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=32;
    QString fileName=exe_path+"conv2d_17_weights.txt";
    QString bias_file=exe_path+"conv2d_17_bias.txt";
    weights_conv2d_17=read_file(fileName);
    bias_conv2d_17=read_file(bias_file);
    l.bias=&bias_conv2d_17[0];
    l.weight=&weights_conv2d_17[0];
    return l;
}
layer initialize_concat_3()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=64;
    return l;
}
layer initialize_conv2d_18_layer()
{
    layer l;
    l.depth=64;
    l.width=3;
    l.height=3;
    l.nfilters=32;
    QString fileName=exe_path+"conv2d_18_weights.txt";
    QString bias_file=exe_path+"conv2d_18_bias.txt";
    weights_conv2d_18=read_file(fileName);
    bias_conv2d_18=read_file(bias_file);
    l.bias=&bias_conv2d_18[0];
    l.weight=&weights_conv2d_18[0];
    return l;
}
layer initialize_conv2d_19_layer()
{
    layer l;
    l.depth=32;
    l.width=3;
    l.height=3;
    l.nfilters=32;
    QString fileName=exe_path+"conv2d_19_weights.txt";
    QString bias_file=exe_path+"conv2d_19_bias.txt";
    weights_conv2d_19=read_file(fileName);
    bias_conv2d_19=read_file(bias_file);
    l.bias=&bias_conv2d_19[0];
    l.weight=&weights_conv2d_19[0];
    return l;
}
layer initialize_up_sampling2d_4()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=32;
    return l;
}
layer initialize_conv2d_20_layer()
{
    layer l;
    l.depth=32;
    l.width=3;
    l.height=3;
    l.nfilters=16;
    QString fileName=exe_path+"conv2d_20_weights.txt";
    QString bias_file=exe_path+"conv2d_20_bias.txt";
    weights_conv2d_20=read_file(fileName);
    bias_conv2d_20=read_file(bias_file);
    l.bias=&bias_conv2d_20[0];
    l.weight=&weights_conv2d_20[0];
    return l;
}
layer initialize_concat_4()
{
    layer l;
    l.depth=0;
    l.width=0;
    l.height=0;
    l.nfilters=32;
    return l;
}
layer initialize_conv2d_21_layer()
{
    layer l;
    l.depth=32;
    l.width=3;
    l.height=3;
    l.nfilters=16;
    QString fileName=exe_path+"conv2d_21_weights.txt";
    QString bias_file=exe_path+"conv2d_21_bias.txt";
    weights_conv2d_21=read_file(fileName);
    bias_conv2d_21=read_file(bias_file);
    l.bias=&bias_conv2d_21[0];
    l.weight=&weights_conv2d_21[0];
    return l;
}
layer initialize_conv2d_22_layer()
{
    layer l;
    l.depth=16;
    l.width=3;
    l.height=3;
    l.nfilters=16;
    QString fileName=exe_path+"conv2d_22_weights.txt";
    QString bias_file=exe_path+"conv2d_22_bias.txt";
    weights_conv2d_22=read_file(fileName);
    bias_conv2d_22=read_file(bias_file);
    l.bias=&bias_conv2d_22[0];
    l.weight=&weights_conv2d_22[0];
    return l;
}
layer initialize_conv2d_23_layer()
{
    layer l;
    l.depth=16;
    l.width=3;
    l.height=3;
    l.nfilters=2;
    QString fileName=exe_path+"conv2d_23_weights.txt";
    QString bias_file=exe_path+"conv2d_23_bias.txt";
    weights_conv2d_23=read_file(fileName);
    bias_conv2d_23=read_file(bias_file);
    l.bias=&bias_conv2d_23[0];
    l.weight=&weights_conv2d_23[0];
    return l;
}
layer initialize_conv2d_24_layer()
{
    layer l;
    l.depth=2;
    l.width=1;
    l.height=1;
    l.nfilters=1;
    QString fileName=exe_path+"conv2d_24_weights.txt";
    QString bias_file=exe_path+"conv2d_24_bias.txt";
    weights_conv2d_24=read_file(fileName);
    bias_conv2d_24=read_file(bias_file);
    l.bias=&bias_conv2d_24[0];
    l.weight=&weights_conv2d_24[0];
    return l;
}
int main(int argc, char const *argv[])
{

    exe_path="/home/nict/plate_finder_section/programs/cuda_c_code/5/weights/";
    cout<<exe_path.toStdString()<<endl;
#if read_from_c1==1
    Mat img(256,256,CV_32F);
    QFile file1(exe_path+"c1.txt"); file1.open(QIODevice::ReadWrite); QTextStream in1(&file1);
    int r=0;
    while (!in1.atEnd()) {
        QString l1=in1.readLine();
        QStringList ll1=l1.split(",");
        for (int c = 0; c < ll1.size(); ++c) {
            img.at<float>(r,c)=ll1[c].toFloat();
        }
        r++;
    }
#else
    Mat img=imread("/home/nict/isv/picture/2/2/44.JPG",IMREAD_GRAYSCALE);//256,256,CV_32FC1);
    cv::resize(img,img,Size(256,256));
    img.convertTo(img,CV_32FC1);
    cv::normalize(img,img,1,0,NORM_MINMAX);
#endif




    layer l_conv2d_1=initialize_conv2d_1_layer();
    layer l_conv2d_2=initialize_conv2d_2_layer();
    layer l_maxp2d_1=initialize_max_pooling2d_1();
    layer l_conv2d_3=initialize_conv2d_3_layer();
    layer l_conv2d_4=initialize_conv2d_4_layer();
    layer l_maxp2d_2=initialize_max_pooling2d_2();
    layer l_conv2d_5=initialize_conv2d_5_layer();
    layer l_conv2d_6=initialize_conv2d_6_layer();
    layer l_maxp2d_3=initialize_max_pooling2d_3();
    layer l_conv2d_7=initialize_conv2d_7_layer();
    layer l_conv2d_8=initialize_conv2d_8_layer();
    layer l_maxp2d_4=initialize_max_pooling2d_4();
    layer l_conv2d_9=initialize_conv2d_9_layer();
    layer l_conv2d_10=initialize_conv2d_10_layer();
    layer l_upsm2d_1=initialize_up_sampling2d_1();
    layer l_conv2d_11=initialize_conv2d_11_layer();
    layer l_concat_1=initialize_concat_1();
    layer l_conv2d_12=initialize_conv2d_12_layer();
    layer l_conv2d_13=initialize_conv2d_13_layer();
    layer l_upsm2d_2=initialize_up_sampling2d_2();
    layer l_conv2d_14=initialize_conv2d_14_layer();
    layer l_concat_2=initialize_concat_2();
    layer l_conv2d_15=initialize_conv2d_15_layer();
    layer l_conv2d_16=initialize_conv2d_16_layer();
    layer l_upsm2d_3=initialize_up_sampling2d_3();
    layer l_conv2d_17=initialize_conv2d_17_layer();
    layer l_concat_3=initialize_concat_3();
    layer l_conv2d_18=initialize_conv2d_18_layer();
    layer l_conv2d_19=initialize_conv2d_19_layer();
    layer l_upsm2d_4=initialize_up_sampling2d_4();
    layer l_conv2d_20=initialize_conv2d_20_layer();
    layer l_concat_4=initialize_concat_4();
    layer l_conv2d_21=initialize_conv2d_21_layer();
    layer l_conv2d_22=initialize_conv2d_22_layer();
    layer l_conv2d_23=initialize_conv2d_23_layer();
    layer l_conv2d_24=initialize_conv2d_24_layer();


    enum LAYER layer_type;
    layer_type=CONV2D_1;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_1);
    layer_type=CONV2D_2;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_2);
    layer_type=MAXP2D_1;LOAD_NEURAL_NETWORK(layer_type,128,128,l_maxp2d_1);
    layer_type=CONV2D_3;LOAD_NEURAL_NETWORK(layer_type,128,128,l_conv2d_3);
    layer_type=CONV2D_4;LOAD_NEURAL_NETWORK(layer_type,128,128,l_conv2d_4);
    layer_type=MAXP2D_2;LOAD_NEURAL_NETWORK(layer_type,64,64,l_maxp2d_2);
    layer_type=CONV2D_5;LOAD_NEURAL_NETWORK(layer_type,64,64,l_conv2d_5);
    layer_type=CONV2D_6;LOAD_NEURAL_NETWORK(layer_type,64,64,l_conv2d_6);
    layer_type=MAXP2D_3;LOAD_NEURAL_NETWORK(layer_type,32,32,l_maxp2d_3);
    layer_type=CONV2D_7;LOAD_NEURAL_NETWORK(layer_type,32,32,l_conv2d_7);
    layer_type=CONV2D_8;LOAD_NEURAL_NETWORK(layer_type,32,32,l_conv2d_8);
    layer_type=MAXP2D_4;LOAD_NEURAL_NETWORK(layer_type,16,16,l_maxp2d_4);
    layer_type=CONV2D_9;LOAD_NEURAL_NETWORK(layer_type,16,16,l_conv2d_9);
    layer_type=CONV2D_10;LOAD_NEURAL_NETWORK(layer_type,16,16,l_conv2d_10);
    layer_type=UPSM2D_1;LOAD_NEURAL_NETWORK(layer_type,32,32,l_upsm2d_1);
    layer_type=CONV2D_11;LOAD_NEURAL_NETWORK(layer_type,32,32,l_conv2d_11);
    layer_type=CONCAT_1;LOAD_NEURAL_NETWORK(layer_type,32,32,l_concat_1);
    layer_type=CONV2D_12;LOAD_NEURAL_NETWORK(layer_type,32,32,l_conv2d_12);
    layer_type=CONV2D_13;LOAD_NEURAL_NETWORK(layer_type,32,32,l_conv2d_13);
    layer_type=UPSM2D_2;LOAD_NEURAL_NETWORK(layer_type,64,64,l_upsm2d_2);
    layer_type=CONV2D_14;LOAD_NEURAL_NETWORK(layer_type,64,64,l_conv2d_14);
    layer_type=CONCAT_2;LOAD_NEURAL_NETWORK(layer_type,64,64,l_concat_2);
    layer_type=CONV2D_15;LOAD_NEURAL_NETWORK(layer_type,64,64,l_conv2d_15);
    layer_type=CONV2D_16;LOAD_NEURAL_NETWORK(layer_type,64,64,l_conv2d_16);
    layer_type=UPSM2D_3;LOAD_NEURAL_NETWORK(layer_type,128,128,l_upsm2d_3);
    layer_type=CONV2D_17;LOAD_NEURAL_NETWORK(layer_type,128,128,l_conv2d_17);
    layer_type=CONCAT_3;LOAD_NEURAL_NETWORK(layer_type,128,128,l_concat_3);
    layer_type=CONV2D_18;LOAD_NEURAL_NETWORK(layer_type,128,128,l_conv2d_18);
    layer_type=CONV2D_19;LOAD_NEURAL_NETWORK(layer_type,128,128,l_conv2d_19);
    layer_type=UPSM2D_4;LOAD_NEURAL_NETWORK(layer_type,256,256,l_upsm2d_4);
    layer_type=CONV2D_20;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_20);
    layer_type=CONCAT_4;LOAD_NEURAL_NETWORK(layer_type,256,256,l_concat_4);
    layer_type=CONV2D_21;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_21);
    layer_type=CONV2D_22;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_22);
    layer_type=CONV2D_23;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_23);
    layer_type=CONV2D_24;LOAD_NEURAL_NETWORK(layer_type,256,256,l_conv2d_24);
    imshow("final_img",img);
    float* output_conv2d_24 = nullptr;


    for (int i = 0; i < 500; ++i) {
        conv2d_1(img.ptr<float>(0),img.cols,img.rows,l_conv2d_1);
        double t1=getTickCount();
        conv2d_2(img.cols,img.rows,l_conv2d_2);
        maxp2d_1(img.cols/2,img.rows/2,l_maxp2d_1);
        conv2d_3(img.cols/2,img.rows/2,l_conv2d_3);
        conv2d_4(img.cols/2,img.rows/2,l_conv2d_4);
        maxp2d_2(img.cols/4,img.rows/4,l_maxp2d_2);
        conv2d_5(img.cols/4,img.rows/4,l_conv2d_5);
        conv2d_6(img.cols/4,img.rows/4,l_conv2d_6);
        maxp2d_3(img.cols/8,img.rows/8,l_maxp2d_3);
        conv2d_7(img.cols/8,img.rows/8,l_conv2d_7);
        conv2d_8(img.cols/8,img.rows/8,l_conv2d_8);
        maxp2d_4(img.cols/16,img.rows/16,l_maxp2d_4);
        conv2d_9(img.cols/16,img.rows/16,l_conv2d_9);
        conv2d_10(img.cols/16,img.rows/16,l_conv2d_10);
        upsample_2d_1(img.cols/8,img.cols/8,l_upsm2d_1);
        conv2d_11(img.cols/8,img.rows/8,l_conv2d_11);
        concat_1(img.cols/8,img.cols/8,l_concat_1);
        conv2d_12(img.cols/8,img.cols/8,l_conv2d_12);
        conv2d_13(img.cols/8,img.cols/8,l_conv2d_13);
        upsample_2d_2(img.cols/4,img.cols/4,l_upsm2d_2);
        conv2d_14(img.cols/4,img.cols/4,l_conv2d_14);
        concat_2(img.cols/4,img.cols/4,l_concat_2);
        conv2d_15(img.cols/4,img.cols/4,l_conv2d_15);
        conv2d_16(img.cols/4,img.cols/4,l_conv2d_16);
        upsample_2d_3(img.cols/2,img.cols/2,l_upsm2d_3);
        conv2d_17(img.cols/2,img.cols/2,l_conv2d_17);
        concat_3(img.cols/2,img.cols/2,l_concat_3);
        conv2d_18(img.cols/2,img.cols/2,l_conv2d_18);
        conv2d_19(img.cols/2,img.cols/2,l_conv2d_19);
        upsample_2d_4(img.cols,img.cols,l_upsm2d_4);
        conv2d_20(img.cols,img.cols,l_conv2d_20);
        concat_4(img.cols,img.cols,l_concat_4);
        conv2d_21(img.cols,img.cols,l_conv2d_21);
        conv2d_22(img.cols,img.cols,l_conv2d_22);
        conv2d_23(img.cols,img.cols,l_conv2d_23);
        conv2d_24(&output_conv2d_24,img.cols,img.cols,l_conv2d_24);
        cout<<"final image without transferring data="<<(getTickCount()-t1)/getTickFrequency()<<endl;

    }

//    cout<<"final image time="<<(getTickCount()-t1)/getTickFrequency()<<endl;
    Mat final_image(256,256,CV_32F,output_conv2d_24);
    cv::normalize(final_image,final_image,255,0,NORM_MINMAX);
    final_image.convertTo(final_image,CV_8U);

    imshow("final_im",final_image);
    waitKey(0);
    //    return 0;
    Size OShape(l_conv2d_2.im_w,l_conv2d_2.im_h);   //output shape
    int ALen=OShape.area();  //array length
    float* output2=(float*)malloc(ALen*sizeof (float));
    vector<Mat> imgs;
//    for (int layer = 0; layer < l_conv2d_2.nfilters; ++layer)
//    {
//        for (int i = 0; i < ALen; ++i)
//        {
//            output2[i]=d_output_conv2[i+ALen*layer];
//        }
//        cv::Mat output_img(OShape.width,OShape.height,CV_32F,output2);
//        cout<<output_img.rowRange(0,10).colRange(0,10)<<endl;
//        imshow("output",output_img);
//        waitKey(0);
//        break;
//        //        imgs.push_back(output_img.clone());
//    }

    //    cout<<imgs[0].rowRange(0,5).colRange(0,5)<<endl;
    //    cout<<imgs[1].rowRange(0,5).colRange(0,5)<<endl;
    //    Mat final_img=0.567117*imgs[0]+0.143773*imgs[1]-0.34974435;
    //    for (int r = 0; r < final_img.rows; ++r) {
    //        for (int c = 0; c < final_img.cols; ++c) {
    //            final_img.at<float>(r,c)=1/(1+cv::exp(-final_img.at<float>(r,c)));
    //        }
    //    }


    //    imshow("final_image",final_img);
    //    waitKey(0);
    //    return 0;

    //    cv::normalize(img,img,255,0,NORM_MINMAX);
    //    img.convertTo(img,CV_8U);


    //    imshow("first_image",img);

    //    Mat added_image(256,256,CV_8U);
    //    for (int r = 0; r < 256; ++r) {
    //        for(int c=0;c<256;c++)
    //        {
    //            if(final_img.at<uchar>(r,c)<200)
    //            {
    //                added_image.at<uchar>(r,c)=(final_img.at<uchar>(r,c)/2+img.at<uchar>(r,c)/2);
    //            }
    //            else
    //            {
    //                added_image.at<uchar>(r,c)=img.at<uchar>(r,c);
    //            }
    //        }
    //    }
    //    imshow("added image",added_image);
    //    waitKey(0);
    //    cout<<final_img.rowRange(0,25).colRange(0,25)<<endl;
    //    QImage imgp((uchar*)final_img.data,final_img.cols,final_img.rows,QImage::Format_Grayscale8);
    //    imgp.save("/home/saeed/CUDA_IN_QT-master/first_layer/1_1.jpg");
    Remove_NN();


}
