Shifting one range to another could come in handy in the data preprocessing task, e.g. shift a GPA of $[0, 10]$ range to $[0, 4]$

Lets denote our first range (or interval) as $[a, b]$ and another as $[c, d]$. Let's recall that an **image interval** is some function $F$ applied to an interval $I$, i.e. $F(I)$. With that in mind, let's first apply the map $t\mapsto t - a$ and get an image interval for the first range: $[0, b-a]$. With that operation we've shifted the left point to $0$. 

Then we apply a scaling mapping to bring the same interval to unit length: $t\mapsto\frac{1}{b-a}\cdot t$. Now we obtain an image interval $[0, 1]$ from the original first range.

Now we need a mapping from the unit interval to our sencond desired range $[c, d]$. Firstly, this involves $t\mapsto(d-c)\cdot t$, which results in the image $[0, d-c]$. Then we do $t\mapsto c+t$, which brings us to $[c, d]$.

Combining these operations we get the following mapping function, which shifts a set $x$ from its range of $[a, b]$ to $[c, d]$: $f(x) = c+(\frac{d-c}{b-a})(x-a)$