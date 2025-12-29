#Compiler to use
CC = g++

#Compiler flags:
# -Wall: Enable all warning messages
#  -Wextra: Enable extra warning flags

CFLAGS = -Wall -Wextra

#DEBUGGING FLAGS USED
DFLAGS = -DDEBUG_MODE -DINHOUSE

#Include directories
INCLUDES = -I/usr/local/zed/include \
	   -I/usr/local/cuda/include \
	   -I../Downloads/teragrid/teraGrid \
	   -I/usr/include/opencv4 \
	   -I../ext/wsServer/include

#Library directories

LDFLAGS = -L/usr/local/zed/lib \
          -L/usr/local/cuda/lib64 \
          -L../ext/wsServer 

#Libraries to link
LIBS = -lsl_zed -lcudart -lcurand -lws -lpthread

# OpenCV flags determined by pkg-config
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

#Source Files

SRCS =  poseestimation.cpp \
	wsServer.c \
	utilities.cpp \
	../Downloads/teragrid/teraGrid/arm_mat_init_f32.c \
	../Downloads/teragrid/teraGrid/arm_fill_f32.c \
	../Downloads/teragrid/teraGrid/teraGrid.c \
	../Downloads/teragrid/teraGrid/arm_mat_trans_f32.c \
	../Downloads/teragrid/teraGrid/arm_mat_mult_f32.c \
	../Downloads/teragrid/teraGrid/arm_mat_sub_f32.c \
	../Downloads/teragrid/teraGrid/arm_copy_f32.c 
#Target Executable name
TARGET = poseestimation


# Generate object file names from source files
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.c=.o)

all:$(TARGET)

#Link the executable

$(TARGET):$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS) $(OPENCV_FLAGS) $(LIBS)

%.o:%.cpp
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@


#Clean up build files
clean:
	rm -f $(TARGET)
