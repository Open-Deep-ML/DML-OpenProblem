
## Understanding Dataset Shuffling

Random shuffling of a dataset is a common preprocessing step in machine learning to ensure that the data is randomly distributed before training a model. This helps to avoid any potential biases that may arise from the order in which data is presented to the model.

### Step-by-Step Method to Shuffle a Dataset

1. **Generate a Random Index Array**  
   Create an array of indices corresponding to the number of samples in the dataset.

2. **Shuffle the Indices**  
   Use a random number generator to shuffle the array of indices.

3. **Reorder the Dataset**  
   Use the shuffled indices to reorder the samples in both \( X \) and \( y \).

### Key Point
This method ensures that the correspondence between \( X \) and \( y \) is maintained after shuffling, preserving the relationship between features and labels.
