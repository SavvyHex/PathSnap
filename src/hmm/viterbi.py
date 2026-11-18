"""HMM/Viterbi route reconstruction.

Hidden states: candidate road segments of each GPS observation.
Emission probabilities: the selected classifier's probability that a
candidate is the correct segment (floored, then log-transformed).
Transition probabilities: exponential penalty on the mismatch between
the GPS displacement of consecutive observations and the candidate-pair
distance (Newson & Krumm style). A small floor probability keeps
disconnected-but-correct transitions reachable.

Edge cases:
- observation with no candidates: the state column is skipped, the
  decoder bridges across the gap, and the observation inherits the
  previous decoded segment (reported as imputed);
- large GPS gaps: the transition penalty saturates, so emissions
  dominate - the decoder degrades gracefully to classifier-only;
- disconnected consecutive segments: allowed at a heavy but finite
  cost, so one bad candidate set cannot derail the whole route.
"""
from dataclasses import dataclass

import numpy as np

from src.config import HMM_BETA, HMM_EPS


@dataclass
class DecodeResult:
    sequence: list          # chosen candidate index per observation (-1 = none)
    imputed: list           # indices of observations with no candidates
    logprob: float


def _log(p):
    return np.log(max(float(p), HMM_EPS))


def viterbi(candidate_probs, gps_dists, cand_dist_fn, beta: float = HMM_BETA) -> DecodeResult:
    """candidate_probs[t]: 1-D array of emission probabilities for
    observation t (empty array when the observation has no candidates).
    gps_dists[t]: great-circle displacement obs t -> t+1 (len T-1).
    cand_dist_fn(t_from, t_to): (n_from, n_to) matrix of candidate-pair
    route distances; np.inf marks disconnected pairs.
    """
    T = len(candidate_probs)
    if T == 0:
        return DecodeResult([], [], 0.0)

    valid = [t for t in range(T) if len(candidate_probs[t]) > 0]
    imputed = [t for t in range(T) if len(candidate_probs[t]) == 0]
    if not valid:
        return DecodeResult([-1] * T, imputed, 0.0)

    deltas, psis = [], []
    deltas.append(np.array([_log(p) for p in candidate_probs[valid[0]]]))
    psis.append(np.zeros(len(deltas[0]), dtype=int))

    for k in range(1, len(valid)):
        t_prev, t = valid[k - 1], valid[k]
        emit = np.array([_log(p) for p in candidate_probs[t]])
        gps_gap = float(sum(gps_dists[t_prev:t]))
        C = np.asarray(cand_dist_fn(t_prev, t), dtype=float)
        C = np.where(np.isinf(C), gps_gap + 10 * beta, C)  # disconnected: heavy finite cost
        trans = -np.abs(gps_gap - C) / max(beta, 1e-9)
        scores = deltas[-1][:, None] + trans
        psis.append(np.argmax(scores, axis=0))
        deltas.append(scores.max(axis=0) + emit)

    back = [0] * len(valid)
    back[-1] = int(np.argmax(deltas[-1]))
    for k in range(len(valid) - 2, -1, -1):
        back[k] = int(psis[k + 1][back[k + 1]])

    sequence = [-1] * T
    for k, t in enumerate(valid):
        sequence[t] = int(back[k])
    for t in imputed:
        sequence[t] = sequence[t - 1] if t > 0 else -1

    return DecodeResult(sequence, imputed, float(np.max(deltas[-1])))
