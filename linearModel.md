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