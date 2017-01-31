// Makes the input tensor sparse along the last dimension
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

REGISTER_OP("MakeSparseGrad")
    .Input("x: T")
    .Input("grad_of_out: T")
    .Input("k: int32")
    .Output("grad_of_inp: T")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

template <typename T>
class MakeSparseGradOp : public OpKernel {
   public:
    explicit MakeSparseGradOp(OpKernelConstruction *context)
        : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensors
        const auto &k_in = context->input(2);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                    errors::InvalidArgument("k must be scalar, got shape ",
                                            k_in.shape().DebugString()));

        int k = k_in.scalar<int32>()();
        OP_REQUIRES(context, k >= 0,
                    errors::InvalidArgument("Need k >= 0, got ", k));

        const Tensor &x_in = context->input(0);

        OP_REQUIRES(context, x_in.dims() >= 1,
                    errors::InvalidArgument("input must be >= 1-D, got shape ",
                                            x_in.shape().DebugString()));

        OP_REQUIRES(
            context, x_in.dim_size(x_in.dims() - 1) >= k,
            errors::InvalidArgument("input must have at least k columns"));

        const Tensor &out_grad = context->input(1);
        OP_REQUIRES(
            context, x_in.shape() == out_grad.shape(),
            errors::InvalidArgument("x and out_grad must have the same shape"));

        // Flattening the input tensor
        const auto &x = x_in.flat_inner_dims<T>();
        const auto &grad_of_out = out_grad.flat_inner_dims<T>();

        const auto num_rows = grad_of_out.dimension(0);
        const auto num_cols = grad_of_out.dimension(1);

        TensorShape output_shape = out_grad.shape();

        // Create an output tensor
        Tensor *inp_grad = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &inp_grad));

        /*
         * Get the top k values along the first dimension for input
         */

        auto grad_of_inp = inp_grad->flat_inner_dims<T>();

        if (k == 0) return;  // Nothing to do

        // Using TopN to get the k max element
        gtl::TopN<std::pair<T, int32>> filter(k);

        // x_sparse.setZero(num_rows, num_cols);

        for (int r = 0; r < num_rows; r++) {
            // Processing a row at a time
            for (int32 c = 0; c < num_cols; c++) {
                // The second element is the negated index, so that lower-index
                // elements
                // are considered larger than higher-index elements in case of
                // ties.
                filter.push(std::make_pair(x(r, c), -c));

                // Initialize output to zero
                grad_of_inp(r, c) = T();
            }

            for (auto top_k_it = filter.unsorted_begin();
                 top_k_it != filter.unsorted_end(); ++top_k_it) {
                grad_of_inp(r, -top_k_it->second) =
                    grad_of_out(r, -top_k_it->second);
            }

            filter.Reset();
        }
    }
};

#define REGISTER_KERNELS_NAME(name, type)                         \
    REGISTER_KERNEL_BUILDER(                                      \
        Name(#name).Device(DEVICE_CPU).TypeConstraint<type>("T"), \
        MakeSparseGradOp<type>)

#define REGISTER_KERNELS(type) REGISTER_KERNELS_NAME(MakeSparseGrad, type);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS_TO_NAME
#undef REGISTER_KERNELS
