#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <QFile>
#include <QTextStream>
#include <iostream>
#include <layer.h>
#include <vector>
#include <thread>

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
    CONV2D_10
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


extern "C" void conv2d_1(float* img_ptr,float** output,int w,int h,layer l);
extern "C" void conv2d_2(float** output,int w,int h,layer l);
extern "C" void maxp2d_1(float** output,int w,int h,layer l);
extern "C" void conv2d_3(float** output,int w,int h,layer l);
extern "C" void conv2d_4(float** output,int w,int h,layer l);
extern "C" void maxp2d_2(float** output, int w, int h, layer l);
extern "C" void conv2d_5(float** output, int w, int h, layer l);
extern "C" void conv2d_6(float** output, int w, int h, layer l);
extern "C" void maxp2d_3(float** output, int w, int h, layer l);
extern "C" void conv2d_7(float** output, int w, int h, layer l);
extern "C" void conv2d_8(float** output, int w, int h, layer l);
extern "C" void maxp2d_4(float** output, int w, int h, layer l);
extern "C" void conv2d_9(float** output, int w, int h, layer l);
extern "C" void conv2d_10(float** output, int w, int h, layer l);

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
int main(int argc, char const *argv[])
{
    exe_path="/home/saeed/CUDA_IN_QT-master/first_layer/weights/";
    cout<<exe_path.toStdString()<<endl;
    Mat img(256,256,CV_32FC1);
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

    cout<<"layer conv2d_2 first weight:"<<l_conv2d_10.weight[0]<<endl;
    cout<<"layer conv2d_2 first bias:"<<l_conv2d_10.bias[0]<<endl;

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
    float* output;
    float* output_conv2d_2;
    float* output_maxp2d_1;
    float* output_conv2d_3;
    float* output_conv2d_4;
    float* output_maxp2d_2;
    float* output_conv2d_5;
    float* output_conv2d_6;
    float* output_maxp2d_3;
    float* output_conv2d_7;
    float* output_conv2d_8;
    float* output_maxp2d_4;
    float* output_conv2d_9;
    float* output_conv2d_10;

    conv2d_1(img.ptr<float>(0),&output,img.cols,img.rows,l_conv2d_1);
    conv2d_2(&output_conv2d_2,img.cols,img.rows,l_conv2d_2);
    maxp2d_1(&output_maxp2d_1,img.cols/2,img.rows/2,l_maxp2d_1);
    conv2d_3(&output_conv2d_3,img.cols/2,img.rows/2,l_conv2d_3);
    conv2d_4(&output_conv2d_4,img.cols/2,img.rows/2,l_conv2d_4);
    maxp2d_2(&output_maxp2d_2,img.cols/4,img.rows/4,l_maxp2d_2);
    conv2d_5(&output_conv2d_5,img.cols/4,img.rows/4,l_conv2d_5);
    conv2d_6(&output_conv2d_6,img.cols/4,img.rows/4,l_conv2d_6);
    maxp2d_3(&output_maxp2d_3,img.cols/8,img.rows/8,l_maxp2d_3);
    conv2d_7(&output_conv2d_7,img.cols/8,img.rows/8,l_conv2d_7);
    conv2d_8(&output_conv2d_8,img.cols/8,img.rows/8,l_conv2d_8);
    maxp2d_4(&output_maxp2d_4,img.cols/16,img.rows/16,l_maxp2d_4);
    conv2d_9(&output_conv2d_9,img.cols/16,img.rows/16,l_conv2d_9);
    conv2d_10(&output_conv2d_10,img.cols/16,img.rows/16,l_conv2d_10);

    Size OShape(l_conv2d_9.im_w,l_conv2d_9.im_h);   //output shape
    int ALen=OShape.area();  //array length
    float* output2=(float*)malloc(ALen*sizeof (float));
    for (int layer = 0; layer < 128; ++layer)
    {
        for (int i = 0; i < ALen; ++i)
        {
            output2[i]=output_conv2d_10[i+ALen*layer];
        }
        cout<<"*****************************************"<<endl;
        cout<<"layer:"<<layer<<endl;
        cout<<"*****************************************"<<endl;
        cv::Mat output_img(OShape.width,OShape.height,CV_32F,output2);
        cout<<output_img.rowRange(0,8).colRange(0,8)<<endl;
    }

    cout<<output_maxp2d_1[128]<<endl;
//    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    Remove_NN();


}
