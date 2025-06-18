
## Understanding MoE Efficiency

The paper *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"* introduces the idea of activating only a few expert networks per input to drastically reduce computation. This is known as **conditional computation**, and it allows models to scale to billions of parameters without significantly increasing cost.

---

### Key Idea

In a **dense layer**, every input goes through the full set of parameters.  
In a **Mixture-of-Experts (MoE)** layer, only $k$ out of $n$ experts are active for each input.

---

### FLOPs Formulas

Let:
- $d_{in}$ = input dimension  
- $d_{out}$ = output dimension  
- $n$ = total experts  
- $k$ = active experts per input  

Then:
- **Dense layer FLOPs**:  
  $$
  \text{FLOPs}_{\text{dense}} = n \cdot d_{in} \cdot d_{out}
  $$
- **MoE layer FLOPs**:  
  $$
  \text{FLOPs}_{\text{moe}} = k \cdot d_{in} \cdot d_{out}
  $$
- **Efficiency gain**:
  $$
  \text{Savings}(\%) = \left( \frac{\text{FLOPs}_{\text{dense}} - \text{FLOPs}_{\text{moe}}}{\text{FLOPs}_{\text{dense}}} \right) \cdot 100
  $$

---

### Example

Suppose:
- $n = 1000$, $k = 2$  
- $d_{in} = d_{out} = 512$  

Then:
- MoE FLOPs = $2 \cdot 512 \cdot 512 = 524,\!288$  
- Full dense (all 1000 experts): $1000 \cdot 512 \cdot 512 = 262,\!144,\!000$  
- Savings:
  $$
  \left( \frac{262,\!144,\!000 - 524,\!288}{262,\!144,\!000} \right) \cdot 100 \approx 99.8\%
  $$

This means the MoE layer uses just **0.2%** of the computation compared to a full dense version — an enormous gain in efficiency.

---

### Summary

By activating only a small number of experts per input, MoE layers reduce computation while maintaining high model capacity. This makes it feasible to train outrageously large models efficiently.
