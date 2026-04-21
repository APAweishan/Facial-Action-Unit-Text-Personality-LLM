# au_selection_sa_masked_multi.py
import argparse
import os
import math
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


AUS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]
AUGMENT_PAD_LENGTH = 6000
BATCH_SIZE = 16

# LSTM 
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.1
MAX_EPOCHS = 5
PATIENCE = 2
LR = 1e-3
GRAD_CLIP_NORM = 1.0

trait_map = {
    'q3': "Honesty-Humility",
    'q4': "Extraversion",
    'q5': "Agreeableness",
    'q6': "Conscientiousness"
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# Dataset & DataLoader
# -------------------------------
class AUDataset(Dataset):
    def __init__(self, data_dir, split, q, au_indices):
        self.q = q
        self.au_indices = au_indices  # list[int]
        self.label_col = trait_map[q]

        label_path = os.path.join(data_dir, f"{split}_data.csv")
        self.label_df = pd.read_csv(label_path)
        self.label_df['id'] = self.label_df['id'].astype(str)
        self.id2label = dict(zip(self.label_df['id'], self.label_df[self.label_col]))

        self.split_csv_dir = os.path.join(data_dir, "FAU_csv", q, f"{split}_csv")
        self.ids = list(self.id2label.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        file_path = os.path.join(self.split_csv_dir, f"{sample_id}.csv")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        au_data = df[[au for au in AUS]].values[:, self.au_indices] if len(self.au_indices) > 0 else np.zeros((len(df), 0))
        au_tensor = torch.tensor(au_data, dtype=torch.float32)
        seq_len = au_tensor.shape[0]

        padded, mask = self.pad_and_mask(au_tensor, AUGMENT_PAD_LENGTH)
        label = torch.tensor(self.id2label[sample_id], dtype=torch.float32)
        return padded, mask, label

    def pad_and_mask(self, x, max_len):
        length = x.shape[0]
        feat_dim = x.shape[1]
        if length >= max_len:
            return x[:max_len], torch.ones(max_len)
        pad = torch.zeros(max_len - length, feat_dim)
        padded = torch.cat([x, pad], dim=0)
        mask = torch.cat([torch.ones(length), torch.zeros(max_len - length)])
        return padded, mask


def collate_fn(batch):
    batch.sort(key=lambda x: x[1].sum(), reverse=True)
    au_datas, masks, labels = zip(*batch)
    au_datas = torch.stack(au_datas)
    masks = torch.stack(masks)
    labels = torch.stack(labels)
    seq_lens = masks.sum(dim=1).long()
    return au_datas, seq_lens, masks, labels


def get_loader(data_dir, split, q):
    def loader(au_indices):
        dataset = AUDataset(data_dir, split, q, au_indices)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == "train"), collate_fn=collate_fn)
    return loader


# -------------------------------
# LSTM model & evaluation
# -------------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, LSTM_HIDDEN, num_layers=LSTM_LAYERS,
            batch_first=True, dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0
        )
        self.linear = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x, seq_lens):
        packed = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        out = self.linear(hn[-1])
        return out.squeeze(-1)


def evaluate_subset_val_mse(au_subset, train_loader_fn, val_loader_fn, device, cache):
    """
    evaluate AU subset corresponding validation set MSE, and use cache to avoid repeated calculations
    assume au_subset is never empty
    """
    key = frozenset(au_subset)
    if key in cache:
        print(f"[Cache hit] AU subset: {sorted(list(au_subset))}")
        return cache[key]

    train_loader = train_loader_fn(list(au_subset))
    val_loader = val_loader_fn(list(au_subset))

    model = SimpleLSTM(input_dim=len(au_subset)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction="mean")

    best_val = float('inf')
    trigger = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, seq_lens, masks, y in train_loader:
            x, seq_lens, y = x.to(device), seq_lens.to(device), y.to(device)
            pred = model(x, seq_lens)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        val_losses = []
        for x, seq_lens, masks, y in val_loader:
            x, seq_lens, y = x.to(device), seq_lens.to(device), y.to(device)
            pred = model(x, seq_lens)
            val_losses.append(criterion(pred, y).item())

        val_mse = float(np.mean(val_losses))
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            trigger = 0
        else:
            trigger += 1
            if trigger >= PATIENCE:
                break

    cache[key] = best_val
    return best_val


# -------------------------------
# Simulated Annealing
# -------------------------------
def neighbor_generate(current, total_au):
    """
    Neighborhood perturbation: Randomly select an AU; if it exists, delete it; if it doesn't, add it. This ensures that an empty set is never created.
    """
    all_idx = list(range(total_au))
    candidate = set(current)

    while True:
        au = random.choice(all_idx)
        if len(candidate) == 1 and au in candidate:
            continue
        if au in candidate:
            candidate.remove(au)
        else:
            candidate.add(au)
        break
    return candidate

def simulated_annealing_once(total_au, train_loader_fn, val_loader_fn, device,
                             init_subset, seed=42,
                             max_iters=50, alpha=0.95,
                             reheating_patience=6, reheating_factor=1.2,
                             cache=None, verbose=True, log_interval=10):
    set_seed(seed)
    if cache is None:
        cache = {}

    current = set(init_subset)
    def evalE(S):
        return evaluate_subset_val_mse(S, train_loader_fn, val_loader_fn, device, cache)

    current_loss = evalE(current)
    best = set(current)
    best_loss = current_loss

    T = max(1e-6, 0.5 * current_loss) if current_loss > 0 else 1.0
    no_improve = 0

    pbar = tqdm(range(1, max_iters + 1), desc=f"[Seed {seed}] SA", ncols=None)

    for it in pbar:
        candidate = neighbor_generate(current, total_au)
        cand_loss = evalE(candidate)
        delta = cand_loss - current_loss

        accept = False
        if delta < 0:
            accept = True
        else:
            prob = math.exp(-delta / max(T, 1e-12))
            if random.random() < prob:
                accept = True

        if accept:
            current, current_loss = candidate, cand_loss

        if current_loss + 1e-12 < best_loss:
            best, best_loss = set(current), current_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= reheating_patience:
            T *= reheating_factor
            no_improve = 0

        T *= alpha

        if verbose and (it % log_interval == 0 or it == 1 or it == max_iters):
            pbar.set_postfix({
                "T": f"{T:.4f}",
                "loss": f"{current_loss:.4f}",
                "best_loss": f"{best_loss:.4f}",
                "subset_size": len(current),
                "subset_preview": ",".join(map(str, list(current)[:5]))
            }, refresh=True)

    pbar.close()
    return best, best_loss, cache

# -------------------------------
# Multiple searches + result integration (Multi-objective + Pareto)
# -------------------------------
def subset_to_vec(subset, n):
    v = np.zeros(n, dtype=np.float32)
    for i in subset:
        v[i] = 1.0
    return v

def jaccard_stability_inverse(s, all_subsets):
    """
    Returns the stability target: the reciprocal of the sum of all Jaccard similarities.
    If sum_sim = 0, it returns a large number. 
    The lower this value, the more stable the subset is (i.e., more similar to other best subsets).
    """
    sum_sim = 0.0
    for other in all_subsets:
        if s is other:
            continue
        inter = len(s & other)
        union = len(s | other)
        sum_sim += inter / union if union > 0 else 0.0
    if sum_sim > 0:
        return 1.0 / sum_sim
    else:
        return 1e6

def pareto_front_minimize(triples_with_payload):
    nd = []
    for i, a in enumerate(triples_with_payload):
        dominated = False
        for j, b in enumerate(triples_with_payload):
            if i == j:
                continue
            if (b[0] <= a[0] and b[1] <= a[1] and b[2] <= a[2]) and (b[0] < a[0] or b[1] < a[1] or b[2] < a[2]):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    return nd

def choose_utopia_solution(pareto_set):
    objs = np.array([[x[0], x[1], x[2]] for x in pareto_set], dtype=np.float64)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    denom = np.where(maxs - mins < 1e-12, 1.0, maxs - mins)
    normed = (objs - mins) / denom
    dists = np.linalg.norm(normed, axis=1)
    idx = int(np.argmin(dists))
    return pareto_set[idx]


def simulated_annealing_multi(total_au, train_loader_fn, val_loader_fn, device,
                              seeds=(42, 3407, 0),
                              init_strategies=("full", "single", "random_k"),
                              max_iters=50, alpha=0.95,
                              reheating_patience=6, reheating_factor=1.2):
    all_best_subsets = []
    all_records = []
    cache = {}

    for seed in seeds:
        set_seed(seed)
        for strat in init_strategies:
            if strat == "full":
                init_subset = list(range(total_au))
            elif strat == "single":
                init_subset = [random.randint(0, total_au - 1)]
            elif strat == "random_k":
                k = random.randint(2, min(16, total_au))
                init_subset = random.sample(range(total_au), k)
            else:
                k = random.randint(2, min(16, total_au))
                init_subset = random.sample(range(total_au), k)

            best_subset, best_loss, cache = simulated_annealing_once(
                total_au, train_loader_fn, val_loader_fn, device,
                init_subset=init_subset, seed=seed,
                max_iters=max_iters, alpha=alpha,
                reheating_patience=reheating_patience, reheating_factor=reheating_factor,
                cache=cache
            )

            all_best_subsets.append(set(best_subset))
            all_records.append((set(best_subset), float(best_loss), int(len(best_subset))))

    triples = []
    for s, mse, size in all_records:
        stability_obj = jaccard_stability_inverse(s, all_best_subsets)
        triples.append((mse, size, stability_obj, s))

    pareto = pareto_front_minimize(triples)
    winner = choose_utopia_solution(pareto)
    return winner, pareto, triples, all_best_subsets


def parse_csv_list(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int_list(value: str):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulated annealing AU subset selection using an LSTM baseline.\n\n"
            "Expected layout under --data-dir:\n"
            "  {train,valid,test}_data.csv\n"
            "  FAU_csv/<q>/{train,valid,test}_csv/<id>.csv\n"
        )
    )
    parser.add_argument("--data-dir", type=str, default="raw_data/FAU_csv", help="Root directory containing labels + FAU_csv tree.")
    parser.add_argument("--q", type=str, default="q4", choices=sorted(trait_map.keys()), help="Question id.")
    parser.add_argument("--device", type=str, default="", help="Force device: cuda or cpu (default: auto).")

    parser.add_argument("--seeds", type=str, default="42,3407,0", help="Comma-separated seeds.")
    parser.add_argument(
        "--init-strategies",
        type=str,
        default="full,single,random_k",
        help="Comma-separated init strategies: full,single,random_k.",
    )
    parser.add_argument("--max-iters", type=int, default=50, help="Simulated annealing iterations per run.")
    parser.add_argument("--alpha", type=float, default=0.95, help="Temperature decay factor.")
    parser.add_argument("--reheating-patience", type=int, default=6)
    parser.add_argument("--reheating-factor", type=float, default=1.2)

    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Epochs for subset evaluation training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pad-length", type=int, default=AUGMENT_PAD_LENGTH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


def main() -> None:
    global MAX_EPOCHS, BATCH_SIZE, AUGMENT_PAD_LENGTH, LR, PATIENCE

    args = parse_args()

    MAX_EPOCHS = int(args.max_epochs)
    BATCH_SIZE = int(args.batch_size)
    AUGMENT_PAD_LENGTH = int(args.pad_length)
    LR = float(args.lr)
    PATIENCE = int(args.patience)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_fn = get_loader(args.data_dir, "train", args.q)
    val_loader_fn = get_loader(args.data_dir, "valid", args.q)

    seeds = tuple(parse_int_list(args.seeds))
    init_strategies = tuple(parse_csv_list(args.init_strategies))

    winner, pareto, triples, best_subsets = simulated_annealing_multi(
        total_au=len(AUS),
        train_loader_fn=train_loader_fn,
        val_loader_fn=val_loader_fn,
        device=device,
        seeds=seeds,
        init_strategies=init_strategies,
        max_iters=args.max_iters,
        alpha=args.alpha,
        reheating_patience=args.reheating_patience,
        reheating_factor=args.reheating_factor,
    )

    best_mse, best_size, stability_obj, best_subset = winner
    best_names = [AUS[i] for i in sorted(best_subset)]

    print("\n====================  Final Results  ====================")
    print("Pareto selected solution (minimum Utopian distance):")
    print(f" - q: {args.q}")
    print(f" - device: {device}")
    print(f" - Val MSE: {best_mse:.6f}")
    print(f" - Subset size: {best_size}")
    print(f" - Stability target (reciprocal): {stability_obj:.6f} (the lower, the more stable)")
    print(f" - AU indices: {sorted(list(best_subset))}")
    print(f" - AU names: {best_names}")

    print("\n====================  Pareto frontier (top-10)  ====================")
    for (mse, size, stab, s) in sorted(pareto, key=lambda x: (x[0], x[1], x[2]))[:10]:
        names = [AUS[i] for i in sorted(s)]
        print(f" * MSE={mse:.6f} | Size={size:2d} | StabilityJ={stab:.6f} | Subset={names}")

    print("\n====================  Multiple best subsets (top-10)  ====================")
    for i, s in enumerate(best_subsets[:10]):
        names = [AUS[j] for j in sorted(s)]
        print(f" {i + 1}: {names}")



if __name__ == "__main__":
    raise SystemExit(main())
    data_dir = "raw_data/FAU_csv"  # <<<--- please set your data directory here, which should contain the CSV files as specified in the code
    q = "q4"       # optional：q3/q4/q5/q6
    assert q in trait_map, f"q must be one of {list(trait_map.keys())}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_fn = get_loader(data_dir, "train", q)
    val_loader_fn = get_loader(data_dir, "valid", q)
    total_au = len(AUS)

    winner, pareto, triples, best_subsets = simulated_annealing_multi(
        total_au=total_au,
        train_loader_fn=train_loader_fn,
        val_loader_fn=val_loader_fn,
        device=device,
        seeds=(42, 3407, 0),
        init_strategies=("full", "single", "random_k"),
        max_iters=50,
        alpha=0.95,
        reheating_patience=6,
        reheating_factor=1.2
    )

    best_mse, best_size, stability_obj, best_subset = winner
    best_names = [AUS[i] for i in sorted(best_subset)]

    print("\n====================  Final Results  ====================")
    print(f"Pareto selected solution (minimum Utopian distance):")
    print(f" - Val MSE: {best_mse:.6f}")
    print(f" - Subset size: {best_size}")
    print(f" - Stability target (reciprocal): {stability_obj:.6f} (the lower, the more stable)")
    print(f" - AU indices: {sorted(list(best_subset))}")
    print(f" - AU names: {best_names}")

    print("\n====================  Pareto frontier (partial display)  ====================")
    for (mse, size, stab, s) in sorted(pareto, key=lambda x: (x[0], x[1], x[2]))[:10]:
        names = [AUS[i] for i in sorted(s)]
        print(f" * MSE={mse:.6f} | Size={size:2d} | StabilityJ={stab:.6f} | Subset={names}")

    print("\n====================  Multiple optimal solutions (partial display)  ====================")
    for i, s in enumerate(best_subsets[:10]):
        names = [AUS[j] for j in sorted(s)]
        print(f" {i+1}: {names}")
