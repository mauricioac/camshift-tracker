all:
	g++ `pkg-config --cflags opencv` -o video main.cpp `pkg-config --libs opencv` -O3 -std=c++11
