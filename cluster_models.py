#!/usr/bin/env python3
"""
cluster_models.py

Three categorical clustering models for GitHub developer profiles produced
by json_to_csv.py.

Models
──────
  LCA  — Latent Class Analysis
           Per-feature categorical EM, MLE estimates.
           Each latent class defines an independent probability distribution
           over every feature column.

  MM   — Multinomial Mixture
           Treats each user as a bag-of-items (languages + libraries pooled
           into one token set) and fits a mixture of multinomials via EM.
           One shared vocabulary; one multinomial φ_k per class.

  NBC  — Naive Bayes Clustering
           Same conditional-independence structure as LCA but uses MAP
           estimation with a Dirichlet(α) prior instead of plain MLE.
           α > 1 acts as Laplace smoothing, preventing zero-probability
           categories and making it more robust on sparse profiles.

Output
──────
  <prefix>_clusters.csv  — original rows + lca_cluster / mm_cluster / nbc_cluster
  <prefix>_clusters.png  — 2-row × 3-col figure:
                             row 1 → cluster size bars
                             row 2 → language-presence heatmap per cluster

Usage
──────
  python cluster_models.py
  python cluster_models.py --input features.csv --output results --k 4
  python cluster_models.py --alpha 2.0 --inits 10
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

LANG_COLS  = [f'top{i}lang' for i in range(1, 6)]
LIB_COLS   = [f'top{i}lib'  for i in range(1, 8)]
FEAT_COLS  = LANG_COLS + LIB_COLS
NONE_TOKEN = '__none__'
EPS        = 1e-300          # log-safety floor


# ═══════════════════════════════════════════════════════════════════════════
# 1 ─ Data loading & encoding
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna(NONE_TOKEN)
    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = NONE_TOKEN
        df[col] = df[col].replace('', NONE_TOKEN)
    return df


def encode_ordinal(df: pd.DataFrame):
    """
    OrdinalEncode every feature column.
    Returns:
        X       – (n, p) int array
        n_cats  – list[int] number of categories per column
        enc     – fitted OrdinalEncoder (for inverse_transform later)
    """
    enc = OrdinalEncoder(dtype=int)
    X   = enc.fit_transform(df[FEAT_COLS].values)
    n_cats = [len(c) for c in enc.categories_]
    return X, n_cats, enc


def build_bow(df: pd.DataFrame):
    """
    Build a bag-of-items count matrix (n × vocab_size).
    Each user's 12 feature values are treated as tokens in a document.
    NONE_TOKEN entries are excluded.
    """
    vocab: dict = {}
    for col in FEAT_COLS:
        for val in df[col].unique():
            if val != NONE_TOKEN and val not in vocab:
                vocab[val] = len(vocab)

    V     = len(vocab)
    X_bow = np.zeros((len(df), V), dtype=np.float64)

    for i, row_vals in enumerate(df[FEAT_COLS].values):
        for val in row_vals:
            if val != NONE_TOKEN and val in vocab:
                X_bow[i, vocab[val]] += 1.0

    return X_bow, vocab


# ═══════════════════════════════════════════════════════════════════════════
# 2 ─ Shared EM utilities
# ═══════════════════════════════════════════════════════════════════════════

def _e_step_categorical(X, log_pi, log_theta):
    """
    Compute soft responsibilities for LCA / NBC.

    Parameters
    ----------
    X          : (n, p) int array
    log_pi     : (K,)  log class priors
    log_theta  : list of p arrays each (K, C_j)

    Returns
    -------
    r  : (n, K)  normalised responsibilities
    ll : float   observed-data log-likelihood
    """
    n, p = X.shape
    K    = len(log_pi)

    log_r = np.tile(log_pi, (n, 1))          # (n, K)
    for j in range(p):
        # log_theta[j][:, X[:, j]] is (K, n)  →  .T gives (n, K)
        log_r += log_theta[j][:, X[:, j]].T

    lse   = log_r.max(axis=1, keepdims=True)
    r     = np.exp(log_r - lse)
    ll    = float((np.log(r.sum(axis=1) + EPS) + lse[:, 0]).sum())
    r    /= r.sum(axis=1, keepdims=True)
    return r, ll


def _m_step_categorical(X, r, n_cats, alpha=0.0):
    """
    M-step for LCA (alpha=0, MLE) or NBC (alpha>0, MAP).

    Returns
    -------
    log_pi    : (K,)
    log_theta : list of (K, C_j) log-probability arrays
    """
    n, p = X.shape
    K    = r.shape[1]
    Nk   = r.sum(axis=0) + EPS                        # (K,)

    # class priors – MAP if alpha>0
    pi = (Nk - 1 + alpha) if alpha > 0 else Nk
    pi = np.maximum(pi, EPS)
    log_pi = np.log(pi / pi.sum())

    log_theta = []
    for j in range(p):
        t = np.zeros((K, n_cats[j]))
        for k in range(K):
            t[k] = np.bincount(X[:, j], weights=r[:, k],
                                minlength=n_cats[j])
        t += alpha                                     # Dirichlet pseudo-counts
        t  = np.maximum(t, EPS)
        t /= t.sum(axis=1, keepdims=True)
        log_theta.append(np.log(t))

    return log_pi, log_theta


# ═══════════════════════════════════════════════════════════════════════════
# 3 ─ Model 1: LCA  (Latent Class Analysis, MLE)
# ═══════════════════════════════════════════════════════════════════════════

def lca_fit(X, n_cats, K, n_init=8, max_iter=400, tol=1e-6, seed=42):
    """
    Fit LCA via EM with multiple random restarts.

    Each latent class k defines an independent categorical distribution
    over every feature j. Parameters θ_kjc = P(X_j = c | Z = k) are
    estimated by MLE (no prior).

    Returns
    -------
    labels : (n,) int array of hard cluster assignments
    resp   : (n, K) soft responsibilities
    ll     : float best observed log-likelihood
    """
    n, p = X.shape
    rng  = np.random.RandomState(seed)
    best = (-np.inf, None, None)

    for _ in range(n_init):
        # Random Dirichlet initialisation
        log_pi    = np.log(rng.dirichlet(np.ones(K)) + EPS)
        log_theta = [np.log(rng.dirichlet(np.ones(c), size=K) + EPS)
                     for c in n_cats]

        prev_ll = -np.inf
        for it in range(max_iter):
            r, ll     = _e_step_categorical(X, log_pi, log_theta)
            log_pi, log_theta = _m_step_categorical(X, r, n_cats, alpha=0.0)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best[0]:
            best = (ll, r.argmax(axis=1), r)

    return best[1], best[2], best[0]


# ═══════════════════════════════════════════════════════════════════════════
# 4 ─ Model 2: Multinomial Mixture  (bag-of-items, MLE)
# ═══════════════════════════════════════════════════════════════════════════

def mm_fit(X_bow, K, n_init=8, max_iter=400, tol=1e-6, seed=42):
    """
    Fit a Mixture of Multinomials via EM.

    Each user is a "document" of items drawn i.i.d. from one multinomial
    per class.  This model pools all 12 feature slots into a single
    vocabulary and ignores positional order (top-1 vs top-5, etc.).

    Returns
    -------
    labels : (n,) int
    resp   : (n, K) float
    ll     : float
    """
    n, V = X_bow.shape
    rng  = np.random.RandomState(seed)
    best = (-np.inf, None, None)

    for _ in range(n_init):
        pi  = rng.dirichlet(np.ones(K))                     # (K,)
        phi = rng.dirichlet(np.ones(V), size=K) + EPS       # (K, V)
        phi /= phi.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for _ in range(max_iter):
            # E-step: log P(doc_i | Z=k) = X_bow @ log(phi_k)
            log_lik = X_bow @ np.log(phi).T                  # (n, K)
            log_r   = log_lik + np.log(pi + EPS)             # (n, K)
            lse     = log_r.max(axis=1, keepdims=True)
            r       = np.exp(log_r - lse)
            ll      = float((np.log(r.sum(axis=1) + EPS) + lse[:, 0]).sum())
            r      /= r.sum(axis=1, keepdims=True)

            # M-step
            Nk   = r.sum(axis=0) + EPS
            pi   = Nk / Nk.sum()
            phi  = r.T @ X_bow + EPS                         # (K, V)
            phi /= phi.sum(axis=1, keepdims=True)

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best[0]:
            best = (ll, r.argmax(axis=1), r)

    return best[1], best[2], best[0]


# ═══════════════════════════════════════════════════════════════════════════
# 5 ─ Model 3: Naive Bayes Clustering  (MAP, Dirichlet prior)
# ═══════════════════════════════════════════════════════════════════════════

def nbc_fit(X, n_cats, K, alpha=1.0, n_init=8, max_iter=400, tol=1e-6, seed=42):
    """
    Fit an unsupervised Naive Bayes model via EM with MAP estimation.

    Structurally identical to LCA but the M-step adds α pseudo-counts
    (Dirichlet prior) to every category count before normalising.
    α = 1 → Laplace (add-one) smoothing.
    α > 1 → stronger regularisation; flatter, more uniform class-conditionals.

    This prevents zero-probability categories (common when profiles are
    sparse) and biases estimates toward more uniform distributions,
    mimicking the Naive Bayes "naive" assumption that all outcomes are
    plausible a-priori.

    Returns
    -------
    labels : (n,) int
    resp   : (n, K) float
    ll     : float
    """
    n, p = X.shape
    rng  = np.random.RandomState(seed)
    best = (-np.inf, None, None)

    for _ in range(n_init):
        log_pi    = np.log(rng.dirichlet(np.ones(K)) + EPS)
        log_theta = [np.log(rng.dirichlet(np.ones(c), size=K) + EPS)
                     for c in n_cats]

        prev_ll = -np.inf
        for _ in range(max_iter):
            r, ll         = _e_step_categorical(X, log_pi, log_theta)
            log_pi, log_theta = _m_step_categorical(X, r, n_cats, alpha=alpha)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best[0]:
            best = (ll, r.argmax(axis=1), r)

    return best[1], best[2], best[0]


# ═══════════════════════════════════════════════════════════════════════════
# 6 ─ Visualisation
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
]


def _lang_presence(df: pd.DataFrame, labels: np.ndarray, K: int):
    """
    Build a (K × n_langs) matrix of language-presence rates per cluster.
    A language "is present" for a user if it appears in any top-N lang slot.
    """
    all_langs = sorted({
        v for col in LANG_COLS
        for v in df[col].unique()
        if v != NONE_TOKEN
    })
    if not all_langs:
        return np.zeros((K, 1)), ['(none)']

    presence = np.zeros((K, len(all_langs)))
    for k in range(K):
        mask = labels == k
        n_k  = max(mask.sum(), 1)
        sub  = df[mask][LANG_COLS]
        for li, lang in enumerate(all_langs):
            presence[k, li] = (sub == lang).values.sum() / n_k

    return presence, all_langs


def plot_results(df, results, K, output_path):
    """
    results : dict  model_name → {'labels': array, 'll': float}
    """
    n_models = len(results)
    fig = plt.figure(figsize=(6 * n_models, 11))
    fig.patch.set_facecolor('#F8F8F8')

    outer = fig.add_gridspec(2, n_models, hspace=0.42, wspace=0.35,
                             top=0.91, bottom=0.06, left=0.06, right=0.97)

    fig.suptitle('Developer Profile Clustering — Model Comparison',
                 fontsize=15, fontweight='bold', y=0.97)

    colors = PALETTE[:K]

    for col, (name, res) in enumerate(results.items()):
        labels  = res['labels']
        ll      = res['ll']
        counts  = np.bincount(labels, minlength=K)
        presence, all_langs = _lang_presence(df, labels, K)

        # ── Row 0: cluster size bar chart ────────────────────────────────
        ax0 = fig.add_subplot(outer[0, col])
        bars = ax0.bar(
            [f'C{k}' for k in range(K)], counts,
            color=colors, edgecolor='white', linewidth=0.8, zorder=3
        )
        ax0.set_facecolor('#F0F0F0')
        ax0.grid(axis='y', color='white', linewidth=1.2, zorder=0)
        ax0.set_title(f'{name}\nlog-lik = {ll:,.1f}',
                      fontsize=11, fontweight='bold', pad=8)
        ax0.set_ylabel('Users', fontsize=9)
        ax0.set_xlabel('Cluster', fontsize=9)
        ax0.tick_params(labelsize=9)
        ax0.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for bar, cnt in zip(bars, counts):
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

        # ── Row 1: language-presence heatmap ─────────────────────────────
        ax1 = fig.add_subplot(outer[1, col])

        # Keep only the top-10 languages by total presence
        top_n  = min(10, len(all_langs))
        totals = presence.sum(axis=0)
        top_idx = np.argsort(totals)[::-1][:top_n]
        pres_top   = presence[:, top_idx]
        top_labels = [all_langs[i] for i in top_idx]

        im = ax1.imshow(
            pres_top, aspect='auto', cmap='YlOrRd',
            vmin=0, vmax=max(pres_top.max(), 1e-3)
        )
        ax1.set_xticks(range(top_n))
        ax1.set_xticklabels(top_labels, rotation=40, ha='right', fontsize=7.5)
        ax1.set_yticks(range(K))
        ax1.set_yticklabels([f'Cluster {k}' for k in range(K)], fontsize=8)
        ax1.set_title('Language Presence per Cluster', fontsize=10, pad=6)

        # Annotate cells
        for r_idx in range(K):
            for c_idx in range(top_n):
                val = pres_top[r_idx, c_idx]
                if val > 0.01:
                    ax1.text(c_idx, r_idx, f'{val:.2f}',
                             ha='center', va='center', fontsize=6.5,
                             color='black' if val < 0.6 else 'white')

        cb = fig.colorbar(im, ax=ax1, shrink=0.75, pad=0.03)
        cb.set_label('Presence rate', fontsize=7)
        cb.ax.tick_params(labelsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot  → {os.path.abspath(output_path)}")


# ═══════════════════════════════════════════════════════════════════════════
# 7 ─ Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run(input_path, output_prefix, K, alpha, n_init, seed):

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {input_path}")
    df = load_data(input_path)
    n  = len(df)
    print(f"  {n} users, {len(FEAT_COLS)} feature columns")

    if n < K:
        sys.exit(f"[ERROR] Only {n} users but K={K} clusters requested. "
                 f"Reduce --k to at most {n}.")

    # ── Encode ────────────────────────────────────────────────────────────
    X_cat, n_cats, enc = encode_ordinal(df)
    X_bow, vocab       = build_bow(df)
    print(f"  Categorical features: {len(n_cats)} cols, "
          f"vocab sizes = [{min(n_cats)}–{max(n_cats)}]")
    print(f"  Bag-of-items vocab : {len(vocab)} tokens")

    # ── Fit models ────────────────────────────────────────────────────────
    results = {}

    print(f"\n[1/3] LCA  (K={K}, {n_init} restarts, MLE)...")
    lca_labels, lca_resp, lca_ll = lca_fit(
        X_cat, n_cats, K, n_init=n_init, seed=seed)
    results['LCA'] = {'labels': lca_labels, 'll': lca_ll}
    print(f"       log-lik = {lca_ll:,.2f}  |  sizes = "
          f"{np.bincount(lca_labels, minlength=K).tolist()}")

    print(f"\n[2/3] Multinomial Mixture  (K={K}, {n_init} restarts, MLE)...")
    mm_labels, mm_resp, mm_ll = mm_fit(
        X_bow, K, n_init=n_init, seed=seed)
    results['Multinomial\nMixture'] = {'labels': mm_labels, 'll': mm_ll}
    print(f"       log-lik = {mm_ll:,.2f}  |  sizes = "
          f"{np.bincount(mm_labels, minlength=K).tolist()}")

    print(f"\n[3/3] NB Clustering  (K={K}, {n_init} restarts, α={alpha})...")
    nbc_labels, nbc_resp, nbc_ll = nbc_fit(
        X_cat, n_cats, K, alpha=alpha, n_init=n_init, seed=seed)
    results['Naive Bayes\nClustering'] = {'labels': nbc_labels, 'll': nbc_ll}
    print(f"       log-lik = {nbc_ll:,.2f}  |  sizes = "
          f"{np.bincount(nbc_labels, minlength=K).tolist()}")

    # ── Output CSV ────────────────────────────────────────────────────────
    csv_path  = f"{output_prefix}_clusters.csv"
    df_out    = df.copy()
    df_out['lca_cluster'] = lca_labels
    df_out['mm_cluster']  = mm_labels
    df_out['nbc_cluster'] = nbc_labels

    # Also attach soft probabilities for each model
    for k in range(K):
        df_out[f'lca_prob_c{k}'] = np.round(lca_resp[:, k], 4)
        df_out[f'mm_prob_c{k}']  = np.round(mm_resp[:, k],  4)
        df_out[f'nbc_prob_c{k}'] = np.round(nbc_resp[:, k], 4)

    df_out.to_csv(csv_path, index=False)
    print(f"\n  CSV   → {os.path.abspath(csv_path)}")

    # ── Output plot ───────────────────────────────────────────────────────
    plot_path = f"{output_prefix}_clusters.png"
    print(f"\nGenerating plot...")
    plot_results(df, results, K, plot_path)

    print(f"\n{'═'*55}")
    print(f"  Done.")
    print(f"  CSV   : {csv_path}")
    print(f"  Plot  : {plot_path}")
    print(f"{'═'*55}\n")

    return df_out


# ═══════════════════════════════════════════════════════════════════════════
# 8 ─ CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Categorical clustering of developer profiles (LCA / MM / NBC).'
    )
    p.add_argument('--input',  '-i', default='Profile_data.csv',
                   help='Input CSV from json_to_csv.py. Default: Profile_data.csv')
    p.add_argument('--output', '-o', default='./Clusters/results',
                   help='Output file prefix. Default: results '
                        '(produces results_clusters.csv / results_clusters.png)')
    p.add_argument('--k', '-k', type=int, default=11,
                   help='Number of clusters. Default: 11')
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Dirichlet prior strength for NBC (≥0). '
                        '0 = MLE like LCA, 1 = Laplace, >1 = stronger smoothing. '
                        'Default: 1.0')
    p.add_argument('--inits', type=int, default=8,
                   help='Random restarts per model. Default: 8')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed. Default: 42')
    args = p.parse_args()

    run(
        input_path    = args.input,
        output_prefix = args.output,
        K             = args.k,
        alpha         = args.alpha,
        n_init        = args.inits,
        seed          = args.seed,
    )


if __name__ == '__main__':
    main()
