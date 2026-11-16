For a linear model y = wx (with no bias), the relationship is directly proportional. Looking at your data:

- (0,0), (1,2), (2,4), (3,6), (4,8)

You can see y = 2x exactly, so **w = 2**.

## How to compute the initial w

### 1. Direct formula (for y = wx)
$$w = \frac{\sum x_i y_i}{\sum x_i^2}$$

Compute:
- Σxᵢyᵢ = 0·0 + 1·2 + 2·4 + 3·6 + 4·8 = 0 + 2 + 8 + 18 + 32 = **60**
- Σxᵢ² = 0 + 1 + 4 + 9 + 16 = **30**

$$w = \frac{60}{30} = 2$$

### 2. For full linear model y = wx + b (least squares)
$$w = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$$

- n = 5, Σx = 10, Σy = 20, Σxy = 60, Σx² = 30

$$w = \frac{5(60) - (10)(20)}{5(30) - 10^2} = \frac{300 - 200}{150 - 100} = \frac{100}{50} = 2$$

$$b = \frac{\sum y - w\sum x}{n} = \frac{20 - 2\cdot10}{5} = 0$$

### 3. Gradient descent starting from w₀ = 0
Update rule: w ← w − α·∂L/∂w, with L = (1/2n)Σ(wxᵢ − yᵢ)²

∂L/∂w = (1/n)Σ(wxᵢ − yᵢ)·xᵢ

Starting at w₀ = 0, with α = 0.01:
- Gradient = (1/5)[(0−0)·0 + (0−2)·1 + (0−4)·2 + (0−6)·3 + (0−8)·4] = (1/5)(0 −2 −8 −18 −32) = −12
- w₁ = 0 − 0.01·(−12) = **0.12**

It will converge toward 2 over many iterations.

**In short:** For this clean dataset the "initial w" from a closed-form fit is **w = 2** (and b = 0). If you mean "initial weight before training" in gradient descent, you typically start with w₀ = 0 (or small random value) and let training find w = 2.


# MSE (Mean Squared Error) for your dataset

## Definition
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - wx_i)^2$$

(For y = wx, no bias.)

## With the correct w = 2
Every prediction is exact, so:

| x | y | ŷ = 2x | (y−ŷ)² |
|---|---|--------|--------|
| 0 | 0 | 0 | 0 |
| 1 | 2 | 2 | 0 |
| 2 | 4 | 4 | 0 |
| 3 | 6 | 6 | 0 |
| 4 | 8 | 8 | 0 |

$$\text{MSE} = \frac{0+0+0+0+0}{5} = \boxed{0}$$

## MSE as a function of w (useful for gradient descent)

$$\text{MSE}(w) = \frac{1}{n}\sum (wx_i - y_i)^2 = \frac{w^2\sum x_i^2 - 2w\sum x_iy_i + \sum y_i^2}{n}$$

Plug in sums (Σx² = 30, Σxy = 60, Σy² = 0+4+16+36+64 = 120, n = 5):

$$\text{MSE}(w) = \frac{30w^2 - 120w + 120}{5} = 6w^2 - 24w + 24$$

Check: at w = 2 → 6(4) − 48 + 24 = **0** ✓

## Gradient of MSE (drives the weight update)
$$\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n}\sum (wx_i - y_i)\,x_i = 12w - 24$$

(Note: this is the gradient of MSE, not of ½·MSE. If you use L = ½·MSE the gradient is 6w − 24.)

## Gradient descent example (start w₀ = 0, α = 0.01, using MSE)
- w₀ = 0: grad = 12(0) − 24 = −24 → w₁ = 0 − 0.01(−24) = **0.24**
- w₁ = 0.24: grad = 12(0.24) − 24 = −21.12 → w₂ = 0.24 + 0.2112 = **0.4512**
- ... converges to **w = 2**, where MSE = 0.

## Key point
MSE is minimized exactly at w = 2 (the value you found from the closed-form fit), and MSE = 0 because the data is perfectly linear through the origin. The gradient of MSE is zero at w = 2: 12(2) − 24 = 0. ✅

If you'd instead like the **½·MSE** convention (common in ML), just halve the values: L(w) = 3w² − 12w + 12, gradient = 6w − 12, still minimized at w = 2 with L = 0.