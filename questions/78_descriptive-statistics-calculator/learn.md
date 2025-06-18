## Understanding Descriptive Statistics

Descriptive statistics provide a summary of data through various measures and help understand the basic structure and distribution of data.

### Key Metrics

Descriptive statistics cover several key metrics, including mean, median, mode, variance, standard deviation, percentiles, quartiles, and interquartile range.

- **Mean**: The average of all values in the dataset, calculated by summing all the values and dividing by the count of values.
- **Median**: The middle value when the data is sorted. If there is an even number of values, the median is the average of the two middle values.
- **Mode**: The value that occurs most frequently in the dataset.
- **Variance**: A measure of how much the values in the dataset deviate from the mean, calculated as:
  $$
  \text{Variance} = \frac{\sum (x_i - \bar{x})^2}{N}
  $$
  where \(x_i\) are the data points, \(\bar{x}\) is the mean, and \(N\) is the number of data points.
- **Standard Deviation**: The square root of the variance, calculated as:
  $$
  \text{Standard Deviation} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N}}
  $$
- **Percentiles and Quartiles**: Percentiles divide the data into 100 equal parts, while quartiles divide the data into four equal parts. The 25th, 50th (median), and 75th percentiles are the common quartiles.
- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles, calculated as:
  $$
  \text{IQR} = Q_3 - Q_1
  $$
  where \(Q_3\) is the 75th percentile and \(Q_1\) is the 25th percentile.

### Example Calculation

Given data:

Data set: \([12, 15, 12, 18, 19, 17, 15, 14, 16, 18]\)

- **Mean**:
  $$
  \text{Mean} = \frac{\sum x_i}{N} = \frac{156}{10} = 15.6
  $$
- **Median**:
  $$
  \text{Median} = 15.5
  $$
- **Mode**:
  $$
  \text{Mode} = 12, 15, 18
  $$
- **Variance**:
  $$
  \text{Variance} = \frac{\sum (x_i - 15.6)^2}{9} \approx 6.93
  $$
- **Standard Deviation**:
  $$
  \text{Standard Deviation} = \sqrt{6.93} \approx 2.63
  $$
- **Percentiles**:
  $$
  25\% \text{ Percentile} = 13.5, \quad 50\% \text{ Percentile} = 15.5 ,\quad 75\% \text{ Percentile} = 17.5
  $$
- **Interquartile Range**:
  $$
  \text{IQR} = 17.5 - 13.5 = 4
  $$

### Applications

Descriptive statistics are widely used in:

- Data Analysis
- Exploratory Data Analysis (EDA)
- Understanding Data Distributions
- Feature Engineering
- Identifying Outliers

These metrics are foundational in data science, helping to interpret data and prepare it for further analysis in machine learning.
