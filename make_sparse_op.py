# Python library for make_sparse operation

import os.path

import tensorflow as tf

from tensorflow.python.framework import ops

# Getting the functions from the shared library
_make_sparse_module = tf.load_op_library(os.path.join(
    tf.resource_loader.get_data_files_path(), 'make_sparse.so'))
make_sparse = _make_sparse_module.make_sparse

_make_sparse_grad_module = tf.load_op_library(os.path.join(
    tf.resource_loader.get_data_files_path(), 'make_sparse_grad.so'))
make_sparse_grad = _make_sparse_grad_module.make_sparse_grad


# Defining the gradients
ops.NotDifferentiable("MakeSparseGrad")


@ops.RegisterGradient("MakeSparse")
def _make_sparse_grad(op, grad):
    """The gradients for 'make_sparse'.

    Args:
        op: The 'make_sparse' operation that we are differentiating, which
            we can use to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of 'make_sparse' op.

    Returns:
        Gradients with respect to the inputs of 'make_sparse'
    """
    x_in = op.inputs[0]
    k = op.inputs[1]

    return [make_sparse_grad(x_in, grad, k), None]
