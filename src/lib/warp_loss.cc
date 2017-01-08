#include "tensorflow/core/framework/op.h"
#include <math.h>
#include<limits.h>
#include<stdio.h>

REGISTER_OP("WarpLoss")
  .Attr("T: {float, double}")
  .Attr("target_dim: int")
  .Attr("use_clean: bool")
  .Input("scores: T")
  .Input("targets: T")
  .Input("correlation_matrix: T")
  .Input("clean: bool")
  .Output("cost: T");

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

double L(int k){
  double loss = 0;
  int i ; 
  for(i = 1; i <= k; i++){
    loss += 1.0/i;
  }   
  return loss;
}

template <typename T>
class WarpLossOp : public OpKernel {
  public:
    explicit WarpLossOp(OpKernelConstruction* context) : OpKernel(context) {
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

      const Tensor& correlation_tensor = context->input(2);
      auto correlation_matrix = correlation_tensor.flat<T>();

      const Tensor& clean_tensor = context->input(3);
      auto clean = clean_tensor.flat<bool>();

      // contant variables used for looping
      const int N = scores.size();
      const int batch_size = N / target_dim;
//      printf("Batch size: %d, target dim: %d, use_clean: %d\n", batch_size, target_dim, use_clean);


//      for(int i = 0; i < batch_size; i++){
//        for (int j = 0; j < target_dim; j++)
//          printf("%lf ", scores(i*batch_size + j));
//        printf("\n");
//      }

      // Create output tensors
      Tensor* cost_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
      &cost_tensor));
      auto average_cost = cost_tensor->flat<T>();

      std::vector<double> cost_per_row;


      // vector to store negative indices in a row
      std::vector<int> nidx; 
      int row_max = 0, max_id = 0;
      double loss;

      for (int i = 0; i < batch_size; i++){
        // find negative indices for this row
        nidx.clear();
        double c = 0;
        double cl = 0;
        for (int j = 0; j < target_dim; j++){
          if (targets(i*target_dim + j) == 0) {
            nidx.push_back(i*target_dim + j);
          }
        }
        
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
        // for every positive target compute cost
        for (int j = 0; j < target_dim; j++){
          if (targets(i*target_dim + j) == 1) {
            // if use_clean is true and this row is not clean and this target is not max target, continue to next iteration of loop
            if (use_clean && (clean(i) == 0) && (j != max_id)) {
//              printf("Skipping %d,%d\n", i, j);
              continue;
            }
            // find violations for a particular target
            int violations = 0;
            for (int n = 0; n < nidx.size(); n++ ){
//              printf("Corellation %f, %f for %d, %d is %f\n", scores(i*target_dim +j), scores(nidx[n]), j, nidx[n]%target_dim, correlation_matrix(j*target_dim + nidx[n]%target_dim));
              cl = fmax(0, correlation_matrix(j*target_dim + nidx[n]%target_dim) - scores(i*target_dim +j) + scores(nidx[n]));
              if (cl > 0){
                violations++;
              }
            }
            // loss
            loss = L(violations);
            for (int n = 0; n < nidx.size(); n++ ){
              cl = loss * fmax(0, correlation_matrix(j*target_dim + nidx[n]%target_dim) - scores(i*target_dim +j) + scores(nidx[n]));
              if (cl > 0){
                c += cl;
              }
            }
          }
        }
        cost_per_row.push_back(c);
      }
      double sum = 0;
      for (int i = 0; i < cost_per_row.size(); i++)
        sum += cost_per_row[i];
      average_cost(0) = sum / batch_size;
    }
    private:
      int target_dim;
      bool use_clean;
};

REGISTER_KERNEL_BUILDER(
    Name("WarpLoss")
    .Device(DEVICE_CPU) 
    .TypeConstraint<float>("T"),
    WarpLossOp<float>);


REGISTER_KERNEL_BUILDER(
    Name("WarpLoss")
    .Device(DEVICE_CPU) 
    .TypeConstraint<double>("T"),
    WarpLossOp<double>);
