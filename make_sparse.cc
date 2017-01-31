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

REGISTER_OP("MakeSparse")
    .Input("x: T")
    .Input("k: int32")
    .Output("x_sparse: T")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename T>
class MakeSparseOp : public OpKernel {
   public:
    explicit MakeSparseOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensors
        const auto &k_in = context->input(1);

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



        // Flattening the input tensor
        const auto &x = x_in.flat_inner_dims<T>();

        const auto num_rows = x.dimension(0);
        const auto num_cols = x.dimension(1);

        TensorShape output_shape = x_in.shape();

        // Create an output tensor
        Tensor *x_out = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, output_shape, &x_out));

        /*
         * Get the top k values along the first dimension for input
         */

        auto x_sparse = x_out->flat_inner_dims<T>();

        if (k == 0) return;  // Nothing to do

        // Using TopN to get the k max element
        gtl::TopN<std::pair<T, int32>> filter(k);

        //x_sparse.setZero(num_rows, num_cols);

        for (int r = 0; r < num_rows; r++) {
            // Processing a row at a time
            for (int32 c = 0; c < num_cols; c++) {
                // The second element is the negated index, so that lower-index
                // elements
                // are considered larger than higher-index elements in case of
                // ties.
                filter.push(std::make_pair(x(r, c), -c));
                x_sparse(r, c) = T();

            }

            for (auto top_k_it = filter.unsorted_begin();
                 top_k_it != filter.unsorted_end(); ++top_k_it) {
                x_sparse(r, -top_k_it->second) = x(r, -top_k_it->second);
            }

            filter.Reset();
        }
    }
};

#define REGISTER_KERNELS_NAME(name, type)                         \
    REGISTER_KERNEL_BUILDER(                                      \
        Name(#name).Device(DEVICE_CPU).TypeConstraint<type>("T"), \
        MakeSparseOp<type>)

#define REGISTER_KERNELS(type) REGISTER_KERNELS_NAME(MakeSparse, type);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS_TO_NAME
#undef REGISTER_KERNELS
