def compute_efficiency(n_experts, k_active, d_in, d_out):
    dense_flops = n_experts * d_in * d_out
    moe_flops = k_active * d_in * d_out
    savings = (dense_flops - moe_flops) / dense_flops * 100
    return round(savings, 1)
