import torch
import torch.nn.functional as F
import math, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 72)
print("  TSO Phase 10 — Réservoir Non-Linéaire (Copie Longue Distance)")
print("=" * 72)

# ---------------------------------------------------------------------------
# ESN Reservoir (continuous activations, echo state property)
# ---------------------------------------------------------------------------
class ESNReservoir(torch.nn.Module):
    def __init__(self, input_dim, res_dim, leak=0.3, spectral_radius=0.9,
                 input_scale=0.5, sparsity=0.1):
        super().__init__()
        self.res_dim = res_dim
        self.leak = leak

        W_in = torch.randn(res_dim, input_dim) * input_scale
        self.register_buffer("W_in", W_in)

        W_rec = torch.randn(res_dim, res_dim) * 1.0
        mask = torch.rand(res_dim, res_dim) > (1 - sparsity)
        W_rec *= mask
        # Normalize spectral radius
        eigs = torch.linalg.eigvals(W_rec)
        rho = eigs.abs().max().item()
        if rho > 0:
            W_rec *= spectral_radius / rho
        self.register_buffer("W_rec", W_rec)

        bias = torch.randn(res_dim) * 0.1
        self.register_buffer("bias", bias)

    def forward(self, inputs):
        batch, seq_len, _ = inputs.shape
        h = torch.zeros(batch, self.res_dim, device=inputs.device)
        states = []
        for t in range(seq_len):
            u = inputs[:, t]
            h = (1 - self.leak) * h + self.leak * torch.tanh(
                h @ self.W_rec.T + u @ self.W_in.T + self.bias
            )
            states.append(h.clone())
        return torch.stack(states, dim=1)

# ---------------------------------------------------------------------------
# Copy task (same as Phase 9)
# ---------------------------------------------------------------------------
def generate_batch(batch_size, seq_len, vocab_size):
    first = torch.randint(0, vocab_size, (batch_size, 1))
    distractor = torch.randint(0, vocab_size, (batch_size, seq_len - 2))
    middle = torch.zeros(batch_size, 1, dtype=torch.long)
    x = torch.cat([first, distractor, middle], dim=1)
    y = first.squeeze(1)
    return x, y

# ---------------------------------------------------------------------------
# Training (readout via gradient descent; reservoir is fixed)
# ---------------------------------------------------------------------------
def train_reservoir(reservoir, readout, vocab_size, seq_len, num_epochs=200,
                    batch_size=64, lr=0.005):
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    accs = []
    for epoch in range(1, num_epochs + 1):
        x, y = generate_batch(batch_size, seq_len, vocab_size)
        x_oh = F.one_hot(x, num_classes=vocab_size).float().to(device)
        y = y.to(device)

        states = reservoir(x_oh)
        final_state = states[:, -1, :]

        logits = readout(final_state)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(-1) == y).float().mean().item()
        accs.append(acc)

        if epoch % 20 == 0 or epoch == 1:
            avg10 = sum(accs[-10:]) / min(len(accs), 10)
            print(f"    Epoch {epoch:4d} | acc={acc:.3f} | avg10={avg10:.3f}")
    final = sum(accs[-10:]) / 10
    return final

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(res_dim, leak, spectral_radius, input_scale, label=""):
    print(f"\n  Reservoir: dim={res_dim}, leak={leak}, ρ={spectral_radius}, {label}")

    for vocab_size, seq_len in [(30, 5), (30, 20), (100, 5), (100, 20)]:
        print(f"\n  --- Vocab={vocab_size}, SeqLen={seq_len} ---")
        reservoir = ESNReservoir(input_dim=vocab_size, res_dim=res_dim,
                                 leak=leak, spectral_radius=spectral_radius,
                                 input_scale=input_scale).to(device)
        readout = torch.nn.Linear(res_dim, vocab_size).to(device)
        acc = train_reservoir(reservoir, readout, vocab_size, seq_len,
                              num_epochs=200, batch_size=64)
        chance = 1.0 / vocab_size
        verdict = "SUCCÈS" if acc > chance * 2 else "ECHEC"
        print(f"  >> acc={acc:.3f} {verdict}  (hasard={chance:.3f})")

t0 = time.time()

# ESN configurations
run(res_dim=500, leak=0.3, spectral_radius=0.9, input_scale=0.3,
    label="ESN Standard")

run(res_dim=1000, leak=0.2, spectral_radius=0.8, input_scale=0.2,
    label="ESN Large / Lent")

run(res_dim=200, leak=0.5, spectral_radius=0.95, input_scale=0.5,
    label="ESN Small / Dynamique Rapide")

run(res_dim=500, leak=0.05, spectral_radius=0.99, input_scale=0.1,
    label="ESN Très Lent (mémoire longue)")

run(res_dim=1000, leak=0.01, spectral_radius=0.5, input_scale=0.05,
    label="ESN Ultra-Lent / Faible Recurrency")

print(f"\n  Temps: {time.time() - t0:.1f}s")
