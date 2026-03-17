Given two vectors `a` and `b` of the same length, compute the projection of `a` onto `b`.

The projection is defined as:

`proj_b(a) = (dot(a, b) / dot(b, b)) * b`

## Input
- `a`: a list of numbers
- `b`: a list of numbers of the same length

## Output
- a list representing the projection of `a` onto `b`

## Notes
- If `b` is the zero vector, raise an error.
- If the vectors do not have the same length, raise an error.