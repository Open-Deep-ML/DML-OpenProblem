## Solution Explanation

Intuition
----

The Intersection over Union (IoU) score is a metric used to evaluate how well a predicted region (for example, a bounding box or segmentation mask) matches the ground truth region.

The intuition is simple:

It measures how much the predicted area overlaps with the true area,
And compares that overlap to the total area covered by both regions combined.
If the prediction perfectly matches the ground truth, IoU = 1 (perfect overlap).
If the prediction does not overlap at all, IoU = 0.

Thus, IoU quantifies spatial similarity between the prediction and the ground truth.

Mathematical definition:
IoU = Area of Overlap/Area of Union


## Step-by-step reasoning: 

1. Identify regions: Determine the predicted area B and ground truth area A.
    
    `INPUTS bboxA and bboxB`

    `bbox = [x1, y1, x2, y2]` 
    
    (x1,y1 -> top left coordinates || x2,y2 -> bottom right coordinates )


2. Compute intersection: Calculate the overlapping area  `|A ∩ B∣`

3. Compute union: Calculate the combined area ∣ 𝐴 ∪ 𝐵 ∣ = ∣ 𝐴 ∣ + ∣ 𝐵 ∣ − ∣ 𝐴 ∩ 𝐵 ∣
` ∣ A∪B ∣ = ∣A∣ + ∣B∣ − ∣ A∩B∣ `

4. Compute IoU: Divide the intersection area by the union area.


Example:
If  
∣A∣=100, ∣B∣=80, and ∣A∩B∣=60,
then:

`∣A∪B∣ = 100 + 80 − 60 = 120`

`IoU = 60/120 = 0.5`

→ The prediction overlaps 50% with the ground truth.


## THE MATH  

The IoU score is calculated as $IoU = \frac{|A \cap B|}{|A \cup B|}$,
where $A$ is the ground truth and $B$ is the predicted region.

where:  
A = Ground truth region   
B = Predicted region   
∣A∩B∣ = Intersection area (common area between A and B)  
∣A∪B∣ = Union area (total area covered by A and B)


--------