all:
	g++ `pkg-config --cflags opencv` -o video main.cpp `pkg-config --libs opencv` -g -std=c++11
