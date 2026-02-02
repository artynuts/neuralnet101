# Neural Network 101 - XOR from Scratch

A simple neural network built from scratch to learn the XOR problem.

## Network Architecture

```
Input Layer (2) → Hidden Layer (4) → Output Layer (1)
```

## Training Data (X)

```
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
```

## Target Output (Y)

```
[[0]
 [1]
 [1]
 [0]]
```

## Initial Weights and Biases

### Weights: Input → Hidden (2×4)

```
[[ 0.49671415 -0.1382643   0.64768854  1.52302986]
 [-0.23415337 -0.23413696  1.57921282  0.76743473]]
```

### Biases: Hidden Layer (1×4)

```
[[0. 0. 0. 0.]]
```

### Weights: Hidden → Output (4×1)

```
[[-0.46947439]
 [ 0.54256004]
 [-0.46341769]
 [-0.46572975]]
```

### Biases: Output Layer (1×1)

```
[[0.]]
```

## Before Training

### Hidden Layer Output (4×4)

```
[[0.5        0.5        0.5        0.5       ]
 [0.44172766 0.44173171 0.829093   0.68296571]
 [0.62168683 0.46548889 0.65648939 0.82098421]
 [0.56526568 0.40796092 0.90263938 0.90808424]]
```

### Network Predictions (4×1)

```
[[0.39459663]
 [0.33849512]
 [0.32609598]
 [0.29208992]]
```

| Input | Expected | Predicted (untrained) |
|-------|----------|----------------------|
| [0,0] | 0        | 0.39                 |
| [0,1] | 1        | 0.34                 |
| [1,0] | 1        | 0.33                 |
| [1,1] | 0        | 0.29                 |

The untrained network predicts ~0.3-0.4 for everything - it hasn't learned XOR yet!

**Initial Loss:** 0.283

## Training Progress

Loss decreases as the network learns over 20,000 epochs:

```
Epoch 0, Loss: 0.28318958906443975
Epoch 1000, Loss: 0.021113218958516283
Epoch 2000, Loss: 0.0027843386080100606
Epoch 3000, Loss: 0.0012599335908539502
Epoch 4000, Loss: 0.0007731558446696055
Epoch 5000, Loss: 0.000544659590570897
Epoch 6000, Loss: 0.00041496642844131835
Epoch 7000, Loss: 0.000332466174610648
Epoch 8000, Loss: 0.0002758369778643973
Epoch 9000, Loss: 0.0002347927456259759
  ...
Epoch 19000, Loss: 0.0000842
```

## After Training

### Final Weights and Biases

#### Weights: Input → Hidden (2×4)

```
[[ 5.94795975  1.92371493  1.5238935   6.41937926]
 [-3.96899568 -4.91313056  3.17750659  6.18801042]]
```

#### Biases: Hidden Layer (1×4)

```
[[ 1.75091895 -0.03758436 -3.28866203 -2.2594114 ]]
```

#### Weights: Hidden → Output (4×1)

```
[[-8.04195652]
 [ 6.16949077]
 [-5.20761089]
 [ 9.16561427]]
```

#### Biases: Output Layer (1×1)

```
[[-1.4434246]]
```

### Hidden Layer Output (4×4)

```
[[0.85206867 0.49060502 0.0359622  0.09454074]
 [0.0981389  0.0070286  0.47223972 0.98070828]
 [0.99954687 0.86831371 0.14619412 0.98463181]
 [0.97656666 0.0462209  0.80419745 0.99996794]]
```

### Network Predictions (4×1)

```
[[0.00644862]
 [0.99180292]
 [0.9898358 ]
 [0.0111749 ]]
```

| Input | Expected | Predicted (trained) |
|-------|----------|---------------------|
| [0,0] | 0        | 0.006               |
| [0,1] | 1        | 0.992               |
| [1,0] | 1        | 0.990               |
| [1,1] | 0        | 0.011               |

**Final Loss:** 0.0000842

The network successfully learned XOR!
