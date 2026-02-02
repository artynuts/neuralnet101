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
