import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    np.random.seed(seed)
    data_dim = 1

    # Initialize generator weights
    w1_g = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    b1_g = np.zeros(hidden_dim)
    w2_g = np.random.normal(0, 0.01, (hidden_dim, data_dim))
    b2_g = np.zeros(data_dim)

    # Initialize discriminator weights
    w1_d = np.random.normal(0, 0.01, (data_dim, hidden_dim))
    b1_d = np.zeros(hidden_dim)
    w2_d = np.random.normal(0, 0.01, (hidden_dim, 1))
    b2_d = np.zeros(1)

    def disc_forward(x):
        h1 = np.dot(x, w1_d) + b1_d
        a1 = relu(h1)
        logit = np.dot(a1, w2_d) + b2_d
        p = sigmoid(logit)
        return p, logit, a1, h1

    def gen_forward(z):
        h1 = np.dot(z, w1_g) + b1_g
        a1 = relu(h1)
        x_gen = np.dot(a1, w2_g) + b2_g
        return x_gen, a1, h1

    for epoch in range(epochs):
        # Sample real data
        x_real = np.random.normal(mean_real, std_real, batch_size)[:, None]
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, _, _ = gen_forward(z)

        # Discriminator forward
        p_real, _, a1_real, h1_real = disc_forward(x_real)
        p_fake, _, a1_fake, h1_fake = disc_forward(x_fake)

        # Discriminator gradients
        grad_logit_real = - (1 - p_real) / batch_size
        grad_a1_real = grad_logit_real @ w2_d.T
        grad_h1_real = grad_a1_real * (h1_real > 0)
        grad_w1_d_real = x_real.T @ grad_h1_real
        grad_b1_d_real = np.sum(grad_h1_real, axis=0)
        grad_w2_d_real = a1_real.T @ grad_logit_real
        grad_b2_d_real = np.sum(grad_logit_real, axis=0)

        grad_logit_fake = p_fake / batch_size
        grad_a1_fake = grad_logit_fake @ w2_d.T
        grad_h1_fake = grad_a1_fake * (h1_fake > 0)
        grad_w1_d_fake = x_fake.T @ grad_h1_fake
        grad_b1_d_fake = np.sum(grad_h1_fake, axis=0)
        grad_w2_d_fake = a1_fake.T @ grad_logit_fake
        grad_b2_d_fake = np.sum(grad_logit_fake, axis=0)

        grad_w1_d = grad_w1_d_real + grad_w1_d_fake
        grad_b1_d = grad_b1_d_real + grad_b1_d_fake
        grad_w2_d = grad_w2_d_real + grad_w2_d_fake
        grad_b2_d = grad_b2_d_real + grad_b2_d_fake

        w1_d -= learning_rate * grad_w1_d
        b1_d -= learning_rate * grad_b1_d
        w2_d -= learning_rate * grad_w2_d
        b2_d -= learning_rate * grad_b2_d

        # Generator update
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, a1_g, h1_g = gen_forward(z)
        p_fake, _, a1_d, h1_d = disc_forward(x_fake)

        grad_logit_fake = - (1 - p_fake) / batch_size
        grad_a1_d = grad_logit_fake @ w2_d.T
        grad_h1_d = grad_a1_d * (h1_d > 0)
        grad_x_fake = grad_h1_d @ w1_d.T

        grad_a1_g = grad_x_fake @ w2_g.T
        grad_h1_g = grad_a1_g * (h1_g > 0)
        grad_w1_g = z.T @ grad_h1_g
        grad_b1_g = np.sum(grad_h1_g, axis=0)
        grad_w2_g = a1_g.T @ grad_x_fake
        grad_b2_g = np.sum(grad_x_fake, axis=0)

        w1_g -= learning_rate * grad_w1_g
        b1_g -= learning_rate * grad_b1_g
        w2_g -= learning_rate * grad_w2_g
        b2_g -= learning_rate * grad_b2_g

    return gen_forward
