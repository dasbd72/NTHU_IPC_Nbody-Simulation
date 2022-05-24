CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS)
TARGET := hw5
SEQUENTIAL := nbody


.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cu
	nvcc $(NVFLAGS) -o hw5 hw5.cu
.PHONY: seq
seq: nbody.cc
	$(CXX) $(CXXFLAGS) -o nbody nbody.cc

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


