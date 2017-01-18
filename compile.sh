#!/bin/bash
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared make_sparse.cc -o make_sparse.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared make_sparse_grad.cc -o make_sparse_grad.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
