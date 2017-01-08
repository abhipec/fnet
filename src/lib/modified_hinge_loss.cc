#include "tensorflow/core/framework/op.h"
#include <math.h>
#include<limits.h>
#include<stdio.h>

REGISTER_OP("ModifiedHingeLoss")
  .Attr("T: {float, double}")
  .Attr("target_dim: int")
  .Attr("use_clean: bool")
  .Input("scores: T")
  .Input("targets: T")
  .Input("clean: bool")
  .Output("cost: T");

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

template <typename T>
class ModifiedHingeLossOp : public OpKernel {
  public:
    explicit ModifiedHingeLossOp(OpKernelConstruction* context) : OpKernel(context) {
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
//      printf("Batch size: %d, target dim: %d, use_clean: %d\n", batch_size, target_dim, use_clean);

//      printf("\nInput Cost\n");
//      for(int i = 0; i < batch_size; i++){
//        for (int j = 0; j < target_dim; j++)
//          printf("%lf ", scores(i*target_dim + j));
//        printf("\n");
//      }

      // Create output tensors
      Tensor* cost_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
      &cost_tensor));
      auto average_cost = cost_tensor->flat<T>();

      std::vector<double> cost_per_row;

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
            }
          }
        }
//        printf("Cost per row: %f\n",cr);
        cost_per_row.push_back(cr);
      }
      double sum = 0;
      for (int i = 0; i < cost_per_row.size(); i++)
        sum += cost_per_row[i];
      average_cost(0) = sum / batch_size;
//      printf("Cost: %lf\n", average_cost(0));
    }
    private:
      int target_dim;
      bool use_clean;
};

REGISTER_KERNEL_BUILDER(
    Name("ModifiedHingeLoss")
    .Device(DEVICE_CPU) 
    .TypeConstraint<float>("T"),
    ModifiedHingeLossOp<float>);


REGISTER_KERNEL_BUILDER(
    Name("ModifiedHingeLoss")
    .Device(DEVICE_CPU) 
    .TypeConstraint<double>("T"),
    ModifiedHingeLossOp<double>);
