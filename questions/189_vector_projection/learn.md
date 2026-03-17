## Solution Explanation

To project a vector `a` onto another vector `b`, we want to keep only the part of `a` that points in the direction of `b`.

In geometry, a vector can often be split into two components:

- one component parallel to `b`
- one component perpendicular to `b`

The projection is the parallel component.

### Intuition

Suppose `a` represents some movement or force, and `b` is the direction we care about.

Projecting `a` onto `b` answers this question:

**How much of vector `a` lies in the direction of vector `b`?**

If `a` points strongly in the same direction as `b`, the projection will be large.  
If `a` is perpendicular to `b`, the projection will be the zero vector.

---

### Mathematical Formula

The projection of `a` onto `b` is:

`proj_b(a) = (dot(a, b) / dot(b, b)) * b`

where:

- `dot(a, b)` is the dot product of `a` and `b`
- `dot(b, b)` is the dot product of `b` with itself

The term

`dot(a, b) / dot(b, b)`

is a scalar that tells us how much of vector `b` we need.

Then we multiply that scalar by `b` to get the projection vector.

---

### Why does this work?

The dot product `dot(a, b)` measures how aligned the two vectors are.

- If the vectors point in similar directions, the dot product is positive.
- If they are perpendicular, the dot product is zero.
- If they point in opposite directions, the dot product is negative.

Dividing by `dot(b, b)` normalizes by the size of `b`, so the formula gives the correct amount of `b` needed for the projection.

---

### Step-by-Step Example

Let:

`a = [3, 4]`, `b = [1, 0]`

#### Step 1: Compute the dot product

`dot(a, b) = (3)(1) + (4)(0) = 3`

#### Step 2: Compute `dot(b, b)`

`dot(b, b) = (1)(1) + (0)(0) = 1`

#### Step 3: Compute the scalar factor

`dot(a, b) / dot(b, b) = 3 / 1 = 3`

#### Step 4: Multiply by `b`

`3 * [1, 0] = [3, 0]`

So the projection of `[3, 4]` onto `[1, 0]` is:

`[3, 0]`

This makes sense because `[1, 0]` points along the x-axis, so we keep only the x-component of `a`.

---

### Another Example

Let:

`a = [2, 2]`, `b = [1, 1]`

#### Step 1: Compute the dot product

`dot(a, b) = (2)(1) + (2)(1) = 4`

#### Step 2: Compute `dot(b, b)`

`dot(b, b) = (1)(1) + (1)(1) = 2`

#### Step 3: Compute the scalar factor

`4 / 2 = 2`

#### Step 4: Multiply by `b`

`2 * [1, 1] = [2, 2]`

So the projection is:

`[2, 2]`

This happens because `a` is already in the same direction as `b`.

---

### Special Cases

#### 1. Orthogonal vectors

If `a` and `b` are perpendicular, then:

`dot(a, b) = 0`

So the projection becomes the zero vector.

Example:

`a = [1, 0]`, `b = [0, 1]`

Then:

`dot(a, b) = 0`

So the projection is the zero vector.

#### 2. Zero vector for `b`

If `b` is the zero vector, then:

`dot(b, b) = 0`

This causes division by zero, which is undefined.

So projection onto the zero vector is not allowed, and the function should raise an error.

---

### Algorithm

To compute the projection of `a` onto `b`:

1. Check that both vectors have the same length.
2. Compute the dot product `dot(a, b)`.
3. Compute the dot product `dot(b, b)`.
4. If `dot(b, b) = 0`, raise an error.
5. Compute the scalar factor `dot(a, b) / dot(b, b)`.
6. Multiply every element of `b` by that scalar.