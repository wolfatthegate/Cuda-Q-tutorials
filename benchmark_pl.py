#!/usr/bin/env python3
import argparse, csv, time, random, math
import networkx as nx
import pennylane as qml
from pennylane import numpy as pnp

# -------------------- Graph / Problems --------------------
def ensure_even_n(n):
    if n < 4: n = 4
    if n % 2 == 1: n += 1
    return n

def rand_3_regular_graph(n, seed=7):
    n = ensure_even_n(n)
    rng = random.Random(seed)
    while True:
        try:
            G = nx.random_regular_graph(3, n, seed=rng.randrange(1 << 30))
            if nx.is_connected(G):
                return G
        except nx.NetworkXError:
            n += 2  # keep even
            n = ensure_even_n(n)

def maxcut_cost_h(G):
    H = 0
    for (u, v) in G.edges():
        H = H + 0.5 * (1 - qml.PauliZ(u) @ qml.PauliZ(v))
    return H

def brute_maxcut_value(G):
    n = G.number_of_nodes()
    if n > 22:
        raise ValueError("Brute force too large; lower n or use --no_bruteforce.")
    best = 0
    for mask in range(1 << n):
        val = 0
        for u, v in G.edges():
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                val += 1
        if val > best:
            best = val
    return best

# -------------------- Toy H2-like Hamiltonian (4 qubits) --------------------
def h2_like_hamiltonian():
    coeffs_ops = [
        (-1.052373245772859, qml.Identity(0)),
        ( 0.397937424843180, qml.PauliZ(0)),
        (-0.397937424843180, qml.PauliZ(1)),
        (-0.011280104256235, qml.PauliZ(2)),
        ( 0.180931199784232, qml.PauliZ(3)),
        ( 0.180931199784232, qml.PauliZ(0) @ qml.PauliZ(1)),
        (-0.011280104256235, qml.PauliZ(0) @ qml.PauliZ(2)),
        ( 0.011280104256235, qml.PauliZ(0) @ qml.PauliZ(3)),
        ( 0.397937424843180, qml.PauliZ(1) @ qml.PauliZ(2)),
        ( 0.011280104256235, qml.PauliZ(1) @ qml.PauliZ(3)),
        (-0.180931199784232, qml.PauliZ(2) @ qml.PauliZ(3)),
        ( 0.120546248542144, qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3)),
        (-0.120546248542144, qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3)),
        (-0.120546248542144, qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3)),
        ( 0.120546248542144, qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3)),
    ]
    coeffs = [float(c) for c, _ in coeffs_ops]
    ops    = [P for _, P in coeffs_ops]
    return qml.Hamiltonian(coeffs, ops), 4

# -------------------- Noise helpers --------------------
def maybe_noise(p_depol, p_ro, n):
    if p_depol > 0.0:
        for i in range(n):
            qml.DepolarizingChannel(p_depol, wires=i)
    if p_ro > 0.0:
        for i in range(n):
            qml.BitFlip(p_ro, wires=i)

def make_device(n, shots, use_mixed):
    name = "default.mixed" if use_mixed else "default.qubit"
    return qml.device(name, wires=n, shots=shots if shots > 0 else None)

# -------------------- Ansatzes --------------------
def qaoa_ansatz(params, G, p_depol, p_ro):
    n = G.number_of_nodes()
    p = params.shape[0] // 2
    betas  = params[:p]
    gammas = params[p:]
    for i in range(n):
        qml.Hadamard(wires=i)
    for l in range(p):
        gamma, beta = gammas[l], betas[l]
        for (u, v) in G.edges():
            qml.IsingZZ(2.0 * gamma, wires=(u, v))
        for i in range(n):
            qml.RX(2.0 * beta, wires=i)
        maybe_noise(p_depol, p_ro, n)

def hea_ansatz(params, n, depth, p_depol, p_ro):
    for d in range(depth):
        base = 2 * n * d
        for i in range(n):
            qml.RY(params[base + 2 * i], wires=i)
            qml.RZ(params[base + 2 * i + 1], wires=i)
        for i in range(n - 1):
            qml.CNOT(wires=(i, i + 1))
        maybe_noise(p_depol, p_ro, n)

# -------------------- Costs / Optim --------------------
def build_cost_fn(H, ansatz, n, dev, *args):
    @qml.qnode(dev, interface="autograd")
    def cost(theta):
        ansatz(theta, *args)
        return qml.expval(H)
    return cost

def adam_minimize(cost, x0, steps=200, lr=0.1):
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    x = x0
    hist = []
    t0 = time.time()
    for s in range(steps + 1):
        e = cost(x)
        hist.append((s, float(e), int((time.time() - t0) * 1000)))
        if s == steps:
            break
        x = opt.step(cost, x)
    return x, hist

def count_ops(ansatz_fn, params, n, *ans_args):
    """
    Build a fresh NO-SHOTS default.qubit device only to materialize the tape,
    then count total ops and 2-qubit ops directly from the tape.
    """
    dev = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(dev)
    def circ(theta):
        ansatz_fn(theta, *ans_args)
        # cheap measurement just to finalize the tape
        return qml.expval(qml.PauliZ(0))

    # run once to build tape (fast on default.qubit without noise)
    _ = circ(params)

    # get the tape across PL versions
    tape = getattr(circ, "tape", None) or getattr(circ, "qtape", None)
    if tape is None:
        return 0, 0

    ops = list(getattr(tape, "operations", []))
    total = len(ops)

    twoq = 0
    twoq_names = {"CNOT", "CZ", "SWAP", "IsingXX", "IsingYY", "IsingZZ", "CRX", "CRY", "CRZ"}
    for op in ops:
        name = getattr(op, "name", "")
        wires = getattr(op, "wires", [])
        # count any op acting on >=2 wires as two-qubit (covers ZZ, CX, etc.)
        if len(wires) >= 2 or name in twoq_names:
            twoq += 1

    return total, twoq

# -------------------- Runner --------------------
def run_maxcut(args):
    G = rand_3_regular_graph(args.n, args.seed)
    H = maxcut_cost_h(G)
    use_mixed = (args.p_depol > 0.0 or args.p_ro > 0.0 or args.shots > 0)

    # QAOA
    dev_q = make_device(G.number_of_nodes(), args.shots, use_mixed)
    cost_qaoa = build_cost_fn(H, qaoa_ansatz, G.number_of_nodes(), dev_q, G, args.p_depol, args.p_ro)
    x0_q = pnp.array(0.1 * pnp.ones(2 * args.p, dtype=float), requires_grad=True)
    xq, hq = adam_minimize(cost_qaoa, x0_q, steps=args.steps, lr=args.lr)
    e_qaoa = float(cost_qaoa(xq))
    ops_q, twoq_q = count_ops(qaoa_ansatz, xq, G.number_of_nodes(), G, args.p_depol, args.p_ro)

    # HEA-VQE
    dev_h = make_device(G.number_of_nodes(), args.shots, use_mixed)
    cost_hea = build_cost_fn(
        H, lambda th, n, d, p1, p2: hea_ansatz(th, n, d, p1, p2),
        G.number_of_nodes(), dev_h, G.number_of_nodes(), args.hea_depth, args.p_depol, args.p_ro
    )
    x0_h = pnp.array(0.1 * pnp.ones(2 * G.number_of_nodes() * args.hea_depth, dtype=float), requires_grad=True)
    xh, hh = adam_minimize(cost_hea, x0_h, steps=args.steps, lr=args.lr)
    e_hea = float(cost_hea(xh))
    ops_h, twoq_h = count_ops(
        lambda th, n, d, p1, p2: hea_ansatz(th, n, d, p1, p2),
        xh, G.number_of_nodes(), G.number_of_nodes(), args.hea_depth, args.p_depol, args.p_ro
    )

    # Optional brute-force ratio
    if not args.no_bruteforce and G.number_of_nodes() <= args.bruteforce_cap:
        best_cut = brute_maxcut_value(G)
        print(f"Approx ratios: QAOA={e_qaoa/best_cut:.4f}, HEA={e_hea/best_cut:.4f}")

    # CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "step", "energy", "elapsed_ms"])
        for s, e, t in hq: w.writerow([f"QAOA(p={args.p})", s, f"{e:.8f}", t])
        for s, e, t in hh: w.writerow([f"HEA(d={args.hea_depth})", s, f"{e:.8f}", t])

    print(f"[MaxCut n={G.number_of_nodes()}] final energies: QAOA={e_qaoa:.6f}, HEA={e_hea:.6f}")
    print(f"Ops (QAOA): total={ops_q}, two-qubit={twoq_q}")
    print(f"Ops (HEA):  total={ops_h}, two-qubit={twoq_h}")
    print(f"CSV -> {args.out}")

def run_h2(args):
    H, n = h2_like_hamiltonian()
    use_mixed = (args.p_depol > 0.0 or args.p_ro > 0.0 or args.shots > 0)

    edges = [(0, 1), (2, 3)]
    def qaoa_h2_ansatz(params, _, p_depol, p_ro):
        p = params.shape[0] // 2
        betas, gammas = params[:p], params[p:]
        for i in range(n): qml.Hadamard(wires=i)
        for l in range(p):
            gamma, beta = gammas[l], betas[l]
            for (u, v) in edges: qml.IsingZZ(2.0 * gamma, wires=(u, v))
            for i in range(n): qml.RX(2.0 * beta, wires=i)
            maybe_noise(p_depol, p_ro, n)

    dev_q = make_device(n, args.shots, use_mixed)
    cost_qaoa = build_cost_fn(H, qaoa_h2_ansatz, n, dev_q, None, args.p_depol, args.p_ro)
    x0_q = pnp.array(0.1 * pnp.ones(2 * args.p, dtype=float), requires_grad=True)
    xq, hq = adam_minimize(cost_qaoa, x0_q, steps=args.steps, lr=args.lr)
    e_qaoa = float(cost_qaoa(xq))
    ops_q, twoq_q = count_ops(qaoa_h2_ansatz, xq, n, None, args.p_depol, args.p_ro)

    dev_h = make_device(n, args.shots, use_mixed)
    cost_hea = build_cost_fn(H, lambda th, _n, d, p1, p2: hea_ansatz(th, _n, d, p1, p2),
                             n, dev_h, n, args.hea_depth, args.p_depol, args.p_ro)
    x0_h = pnp.array(0.1 * pnp.ones(2 * n * args.hea_depth, dtype=float), requires_grad=True)
    xh, hh = adam_minimize(cost_hea, x0_h, steps=args.steps, lr=args.lr)
    e_hea = float(cost_hea(xh))
    ops_h, twoq_h = count_ops(lambda th, _n, d, p1, p2: hea_ansatz(th, _n, d, p1, p2),
                              xh, n, n, args.hea_depth, args.p_depol, args.p_ro)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "step", "energy", "elapsed_ms"])
        for s, e, t in hq: w.writerow([f"QAOA(p={args.p})", s, f"{e:.8f}", t])
        for s, e, t in hh: w.writerow([f"HEA(d={args.hea_depth})", s, f"{e:.8f}", t])

    print(f"[H2-like 4q] final energies: QAOA={e_qaoa:.6f}, HEA={e_hea:.6f}")
    print(f"Ops (QAOA): total={ops_q}, two-qubit={twoq_q}")
    print(f"Ops (HEA):  total={ops_h}, two-qubit={twoq_h}")
    print(f"CSV -> {args.out}")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Original PennyLane benchmark harness (fixed)")
    ap.add_argument("--problem", choices=["maxcut", "h2"], default="maxcut")
    ap.add_argument("--n", type=int, default=12, help="qubits for MaxCut")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p", type=int, default=2, help="QAOA depth")
    ap.add_argument("--hea_depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--p_depol", type=float, default=0.0)
    ap.add_argument("--p_ro", type=float, default=0.0)
    ap.add_argument("--no_bruteforce", action="store_true")
    ap.add_argument("--bruteforce_cap", type=int, default=18)
    ap.add_argument("--out", type=str, default="pl_results.csv")
    args = ap.parse_args()

    if args.problem == "maxcut":
        run_maxcut(args)
    else:
        run_h2(args)

if __name__ == "__main__":
    main()