Implement a function that calculates the disorder in a basket of apples based on their colors, where each apple color is represented by an integer. The disorder must be 0 if all apples are the same color and must increase as the variety of colors increases. In particular:
- [0,0,0,0] should yield 0.
- [1,1,0,0] should have a higher disorder than [0,0,0,0].
- [0,1,2,3] should have a higher disorder than [1,1,0,0].
- [0,0,1,1,2,2,3,3] should have a higher disorder than [0,0,0,0,0,1,2,3].

You may use any method to measure disorder as long as these properties are satisfied.
