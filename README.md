# MakeSparseTF
Tensorflow function to make a tensor sparse (gradients supported)

The ```make_sparse``` op makes the input tensor sparse along its last dimension. ```make_sparse_grad``` op is used to compute the gradients for make_sparse.

# Why the custom function?
There is no inbuilt support for making a tensor sparse in tensorflow . The ````top_k```` op of tensorflow can be used to achieve sparsity but does not support gradient computation. 

This project reuses code for ````top_k```` to achieved the desired operation._

# How to use?
Copy ````make_sparse.so````, ````make_sparse_grad.so````, and ````make_sparse_op.py```` to the working directory of your project.

Then in the python code, import the op
	
		from make_sparse_op import make_sparse
		

The op works as follows
	
		x_sparse = make_sparse(x, k)
		
where,

* ````x```` is input tensor along the last dimension of which the sparsity operation is applied.
* ````k```` is a scalar tensor which specifies the max number of non-zero values to keep (i.e. the desired sparsity).
* ````x_sparse```` is the output which is the sparse version of ````x````.
		
