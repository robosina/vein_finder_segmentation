TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG += qt
CONFIG += c++11
SOURCES += \
    main.cpp
DESTDIR = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
#QMAKE_CXXFLAGS_RELEASE = -03
DISTFILES += \
    conv2d_1.cu \
    conv2d_1.cu
CUDA_SOURCES += conv2d_1.cu \
                layer.c
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS +=-L $$CUDA_DIR/lib -lcudart -lcuda
CUDA_ARCH = sm_21
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output  = ${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

SOURCES +=
#INCLUDEPATH+=/usr/local/cudnn/include
#LIBS+=-L/usr/local/cudnn/lib64 -lcudnn

INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_tracking -lopencv_img_hash -lopencv_bgsegm -lopencv_cudabgsegm -lopencv_hfs -lopencv_cudastereo -lopencv_reg -lopencv_rgbd -lopencv_face -lopencv_photo -lopencv_stereo -lopencv_plot -lopencv_cudacodec -lopencv_line_descriptor -lopencv_cudaoptflow -lopencv_dpm -lopencv_saliency -lopencv_freetype -lopencv_ccalib -lopencv_aruco -lopencv_bioinspired -lopencv_datasets -lopencv_text -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudalegacy -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudawarping -lopencv_surface_matching -lopencv_optflow -lopencv_ximgproc -lopencv_video -lopencv_dnn_objdetect -lopencv_dnn -lopencv_xphoto -lopencv_xobjdetect -lopencv_objdetect -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_cudaarithm -lopencv_fuzzy -lopencv_structured_light -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_phase_unwrapping -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_cudev

HEADERS += \
    layer.h
