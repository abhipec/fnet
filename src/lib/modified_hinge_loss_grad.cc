#include "tensorflow/core/framework/op.h"
#include <math.h>
#include<limits.h>
#include<stdio.h>

REGISTER_OP("ModifiedHingeLossGrad")
  .Attr("T: {float, double}")
  .Attr("target_dim: int")
  .Attr("use_clean: bool")
  .Input("scores: T")
  .Input("targets: T")
  .Input("clean: bool")
  .Output("gradients_scores: T")
  .Output("gradients_targets: T")
  .Output("gradients_clean: bool");

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

template <typename T>
class ModifiedHingeLossGradOp : public OpKernel {
  public:
    explicit ModifiedHingeLossGradOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context,
                             context->GetAttr("target_dim", &target_dim));
      OP_REQUIRES_OK(context,
                             context->GetAttr("use_clean", &use_clean));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensors
      const Tensor& score_tensor = context->input(0);
      auto scores = score_tensor.flat<T>();

      const Tensor& target_tensor = context->input(1);
      auto targets = target_tensor.flat<T>();


      const Tensor& clean_tensor = context->input(2);
      auto clean = clean_tensor.flat<bool>();

      // contant variables used for looping
      const int N = scores.size();
      const int batch_size = N / target_dim;

      // Create output tensors

      Tensor* gradient_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, score_tensor.shape(),
      &gradient_tensor));
      auto gradient = gradient_tensor->flat<T>();

      Tensor* gradient_tensor2 = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, target_tensor.shape(),
      &gradient_tensor2));
      auto gradient2 = gradient_tensor2->flat<T>();

      Tensor* gradient_tensor3 = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({batch_size}),
      &gradient_tensor3));
      auto gradient3 = gradient_tensor3->flat<bool>();

      // set all gradients to zero
      for (int i = 0; i < gradient.size(); i++){
        gradient(i) = 0;
        gradient2(i) = 0;
      }

      int row_max = 0, max_id = 0;
      double cr = 0;
      double s = 0;

      for (int i = 0; i < batch_size; i++){
        cr = 0;
        s = 0;
        if (use_clean) {
          // find max element indices per row from positive targets
          row_max = INT_MIN;
          max_id = 0;
          for ( int j = 0; j < target_dim; j++) {
            if ((scores(i*target_dim + j) > row_max) && (targets(i*target_dim +j) == 1)) {
              row_max = scores(i*target_dim + j);
              max_id = j;
            }
          }
        }
        // loop through every labels
        for (int j = 0; j < target_dim; j++){
          // if labels is +ve
          if (targets(i*target_dim + j) == 1) {
            // if use_clean is true and this row is not clean and this target is not max target,
            // continue to next iteration of loop
            if (use_clean && (clean(i) == 0) && (j != max_id)) {
//              printf("Skipping %d,%d\n", i, j);
              continue;
            }
            // cost
            s = scores(i*target_dim + j);
            if (s >= 1) {
              cr += 0;
            }
            else {
              cr += -s + 1;
              gradient(i*target_dim +j) += (-1.0 / batch_size); 
            }
          }
          // if labels is -ve
          else {
            s = scores(i*target_dim + j);
            if (s <= -1) {
              cr += 0;
            }
            else {
              cr += s + 1;
              gradient(i*target_dim +j) += (1.0 / batch_size); 
            }
          }
        }
      }

//      printf("\nGradient\n");
//      for(int i = 0; i < batch_size; i++){
//        for (int j = 0; j < target_dim; j++)
//          printf("%lf ", gradient(i*target_dim + j));
//        printf("\n");
//      }
    }
    private:
      int target_dim;
      bool use_clean;
};

REGISTER_KERNEL_BUILDER(
    Name("ModifiedHingeLossGrad")
    .Device(DEVICE_CPU) 
    .TypeConstraint<float>("T"),
    ModifiedHingeLossGradOp<float>);


REGISTER_KERNEL_BUILDER(
    Name("ModifiedHingeLossGrad")
    .Device(DEVICE_CPU) 
    .TypeConstraint<double>("T"),
    ModifiedHingeLossGradOp<double>);
