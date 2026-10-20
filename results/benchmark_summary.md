# Ground-truth benchmark results

Predictions per (rho, model): 20 · seed 0

**Trust vs correctness (instance level): r = +0.608**

## Mean Explanation Trust Score (rho x model)

```
model  gradient_boosting  logistic_regression   mlp  random_forest
rho                                                               
0.00                76.2                 93.5  86.3           77.1
0.30                73.3                 94.6  88.3           80.0
0.60                74.1                 95.4  90.0           79.4
0.80                68.6                 92.5  83.4           69.8
0.95                65.9                 95.8  75.7           67.3
```

## True-driver precision (rho x method)

```
method  integrated_gradients   lime  occlusion  shapley  smoothgrad_x_input
rho                                                                        
0.00                   0.822  0.897      0.859    0.900               0.806
0.30                   0.828  0.875      0.828    0.872               0.772
0.60                   0.788  0.838      0.759    0.831               0.747
0.80                   0.775  0.866      0.750    0.806               0.722
0.95                   0.753  0.741      0.612    0.659               0.647
```

## True-driver precision (model x method)

```
method               integrated_gradients   lime  occlusion  shapley  smoothgrad_x_input
model                                                                                   
gradient_boosting                   0.695  0.835      0.702    0.780               0.618
logistic_regression                 0.938  0.935      0.898    0.930               0.938
mlp                                 0.788  0.762      0.712    0.762               0.752
random_forest                       0.752  0.840      0.735    0.782               0.648
```
