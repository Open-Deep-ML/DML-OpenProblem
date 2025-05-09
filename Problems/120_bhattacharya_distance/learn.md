
# Learn Section

## Understanding Bhattacharyya Distance

**Bhattacharyya Distance (BD)** is a concept in statistics used to measure the **similarity** or **overlap** between two probability distributions **P(x)** and **Q(x)** on the same domain **x**.  

This differs from **KL Divergence**, which measures the **loss of information** when projecting one probability distribution onto another (reference distribution).  

### **Bhattacharyya Distance Formula**
The Bhattacharyya distance is defined as:  

$$
BC (P, Q) = \sum \sqrt{P(X) \cdot Q(X)}
$$

$$
BD (P, Q) = -\ln(BC (P, Q))
$$

where **BC (P, Q)** is the **Bhattacharyya coefficient**.  

### **Key Properties**
1. **BD is always non-negative**:  
   $$ BD \geq 0 $$
2. **Symmetric in nature**:  
   $$ BD (P, Q) = BD (Q, P) $$
3. **Applications**:  
   - Risk assessment  
   - Stock predictions  
   - Feature scaling  
   - Classification problems  

### **Example Calculation**
Consider two probability distributions **P(x)** and **Q(x)**:  

$$
P(x) = [0.1, 0.2, 0.3, 0.4], \quad Q(x) = [0.4, 0.3, 0.2, 0.1]
$$

1. **Bhattacharyya Coefficient**:  

$$
BC (P, Q) = \sum \sqrt{P(X) \cdot Q(X)} = 0.8898
$$

2. **Bhattacharyya Distance**:  

$$
BD (P, Q) = -\ln(BC (P, Q)) = -\ln(0.8898) = 0.1166
$$

This illustrates how BD quantifies the **overlap** between two probability distributions.  

    