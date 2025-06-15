## Simple Convolutional 2D Layer

The Convolutional layer is a fundamental component used extensively in Computer Vision tasks. Here are the crucial parameters:

### Parameters
1. **input_matrix**:  
   A 2D NumPy array representing the input data, such as an image. Each element in this array corresponds to a pixel or a feature value in the input space. The dimensions of the input matrix are typically represented as $ \text{height} \times \text{width} $.

2. **kernel**:  
   Another 2D NumPy array representing the convolutional filter. The kernel is smaller than the input matrix and slides over it to perform the convolution operation. Each element in the kernel serves as a weight that modifies the input during convolution. The kernel size is denoted as $ \text{kernel\_height} \times \text{kernel\_width} $.

3. **padding**:  
   An integer specifying the number of rows and columns of zeros added around the input matrix. Padding controls the spatial dimensions of the output, allowing the kernel to process edge elements effectively or to maintain the original input size.

4. **stride**:  
   An integer that represents the number of steps the kernel moves across the input matrix for each convolution. A stride greater than one reduces the output size, as the kernel skips over elements.

### Implementation
1. **Padding the Input**:  
   The input matrix is padded with zeros based on the specified `padding` value. This increases the input size and enables the kernel to cover elements at the borders and corners.

2. **Calculating Output Dimensions**:  
   The height and width of the output matrix are calculated using the following formulas:
   $$
   \text{output\_height} = \left( \frac{\text{input\_height, padded} - \text{kernel\_height}}{\text{stride}} \right) + 1
   $$
   $$
   \text{output\_width} = \left( \frac{\text{input\_width, padded} - \text{kernel\_width}}{\text{stride}} \right) + 1
   $$

3. **Performing Convolution**:
   - A nested loop iterates over each position where the kernel can be applied to the padded input matrix.
   - At each position, a region of the input matrix, matching the size of the kernel, is selected.
   - Element-wise multiplication between the kernel and the input region is performed, followed by summing the results to produce a single value. This value is then stored in the corresponding position of the output matrix.

4. **Output**:  
   The function returns the output matrix, which contains the results of the convolution operation performed across the entire input.
