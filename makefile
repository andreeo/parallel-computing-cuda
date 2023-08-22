# makefile for CUDA programs

CC = nvcc
CFLAGS =
FPFLAGS = --cudart shared
LDFLAGS =
SOURCEFILES =
3RDPARTY =
TARGET = $(target)

ifeq ($(target), $(filter $(target),sum_vectors matrix2dim pi displacer_matrix counting_sort))
	SOURCEFILES = ./src/$(target).cu cuda_device.cpp ./src/common.cpp ./src/kernels.cu
else ifeq ($(target), show_devices)
	SOURCEFILES = $(target).cu cuda_device.cpp
endif

all: check-param $(TARGET)

$(TARGET): $(SOURCEFILES)
	$(CC) $(CFLAGS) $(FPFLAGS) -o $(TARGET) $(SOURCEFILES) $(3RDPARTY) $(LDFLAGS)

clean: check-param
	rm -rf $(TARGET) _* *ptx* checkpoint_files *~ src/*~

indent: indent-format clean

indent-format:
	indent *.cu *.cpp *.h src/*.cu src/*.cpp src/*.h \
		-nbad -bap -nbc -bbo -bl -bli0 -bls -ncdb -nce -cp1 -cs -di2 \
		-ndj -nfc1 -nfca -hnl -i2 -ip5 -lp -pcs -nprs -psl -saf -sai \
		-saw -nsc -nsob -nut

run: check-param 
	./$(TARGET)

check-param:
ifndef target
	@echo ""
	@echo "Usage: make [OPTIONAL] target=<filename>"
	@echo "  where [OPTIONAL] is one of the following:"
	@echo "    all (default) - compile the target file"
	@echo "    clean         - remove all files generated by make"
	@echo "    indent        - format the source code using indent"
	@echo "    run           - run the executable"
	@echo "  where <filename> is the name of the target file to be compiled"
	@echo "  (without the .cu extension)"
	@echo ""
endif