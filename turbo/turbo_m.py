###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from .gp import train_gp
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class TurboM(Turbo1):
    """The TuRBO-m algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points *FOR EACH TRUST REGION* (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    n_trust_regions : Number of trust regions
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo5 = TurboM(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
        turbo5.optimize()  # Run optimization
        X, fX = turbo5.X, turbo5.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):
        self.n_trust_regions = n_trust_regions
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )

        self.succtol = 3
        self.failtol = max(5, self.dim)

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)
        self._processed_counts = np.zeros(self.n_trust_regions, dtype=int)

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next

    def optimize(self):
        """Run the full optimization process."""
        # Create initial points for each TR
        for i in range(self.n_trust_regions):
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.X = np.vstack((self.X, X_init))
            self.fX = np.vstack((self.fX, fX_init))
            self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            self.n_evals += self.n_init

            if self.verbose:
                fbest = fX_init.min()
                print(f"TR-{i} starting from: {fbest:.4}")
                sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # Generate candidates from each TR
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())

                # Don't retrain the model if the training data hasn't changed
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                    X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])

            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:  # Restart trust region if converged
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    X_init = from_unit_cube(X_init, self.lb, self.ub)
                    fX_init = np.array([[self.f(x)] for x in X_init])

                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init

    def propose_next_config(
        self,
        X_obs: np.ndarray,
        fX_obs: np.ndarray,
        idx_obs: np.ndarray | None = None,
        n_suggestions: int | None = None,
    ):
        """
        Propose the next batch of configurations WITHOUT evaluating them.

        Parameters
        ----------
        X_obs : ndarray, shape (N, d)
            All configurations observed so far (original space).
        fX_obs : ndarray, shape (N, 1) or (N,)
            Corresponding objective values (minimization).
        idx_obs : ndarray, shape (N, 1) or (N,), optional
            Trust-region assignment for each row in X_obs. If None, we assume all
            points belong to TR-0 (single-TR usage).
        n_suggestions : int, optional
            How many total points to return across all trust regions. Defaults to self.batch_size.

        Returns
        -------
        X_next : ndarray, shape (n_suggestions, d)
            Next configurations to evaluate (original space).
        idx_next : ndarray, shape (n_suggestions, 1)
            Trust-region index that proposed each configuration.

        Notes
        -----
        - This method NEVER calls self.f; it only proposes points.
        - It uses/updates self.hypers (per-TR cached GP hypers) to avoid retraining
          when a TR didn't receive new data since the last call.
        - Before proposing, update TR lengths/counters using the most recent
          finished observations *per TR*, but only once a TR has >= n_init points.
        - The 'new' observations are those that arrived since the last call to ask()
          (tracked via self._processed_counts).
        """
        assert X_obs.ndim == 2 and X_obs.shape[1] == self.dim, "X_obs must be (N, d)"
        fX_obs = fX_obs.reshape(-1, 1)
        assert X_obs.shape[0] == fX_obs.shape[0], "X_obs and fX_obs must have the same N"

        if idx_obs is None:
            idx_obs = np.zeros((X_obs.shape[0], 1), dtype=int)
        else:
            idx_obs = idx_obs.reshape(-1, 1)
            assert idx_obs.shape[0] == X_obs.shape[0], "idx_obs length must match X_obs"

        n_total = int(self.batch_size if n_suggestions is None else n_suggestions)
        assert n_total >= 1

        # ---------- NEW: update TR state BEFORE proposing ----------
        # We only start adjusting a TR once it has >= n_init observations.
        # We only adjust for observations that are new since our last call.
        for i in range(self.n_trust_regions):
            tr_idx_all = np.where(idx_obs[:, 0] == i)[0]
            cnt_i = len(tr_idx_all)
            prev_cnt = int(self._processed_counts[i])

            # We start accounting from the first index >= n_init
            start_idx = max(prev_cnt, self.n_init)

            if cnt_i > start_idx:
                # The tail since we last processed (or since n_init if first time)
                new_tail_idx = tr_idx_all[start_idx:cnt_i]
                fX_new = fX_obs[new_tail_idx, :]

                # Emulate the original _adjust_length contract using current full history
                _fX_backup, _idx_backup = getattr(self, "fX", None), getattr(self, "_idx", None)
                self.fX = fX_obs  # full history
                self._idx = idx_obs
                self._adjust_length(fX_next=fX_new, i=i)
                self.fX, self._idx = _fX_backup, _idx_backup

                # Force refit next time this TR is used
                self.hypers[i] = {}
                # Mark these as processed
                self._processed_counts[i] = cnt_i

        # ---------- Candidate generation (unchanged) ----------
        X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
        y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))

        for i in range(self.n_trust_regions):
            tr_mask = (idx_obs[:, 0] == i)
            if not np.any(tr_mask):
                X_seed = latin_hypercube(max(self.n_init, 2 * self.dim), self.dim)
                y_seed = 1e9 * np.ones(X_seed.shape[0])
                X_unit = X_seed
                fX_std = y_seed
                n_training_steps = self.n_training_steps
            else:
                X_i = deepcopy(X_obs[tr_mask, :])
                fX_i = deepcopy(fX_obs[tr_mask, 0].ravel())

                X_unit = to_unit_cube(X_i, self.lb, self.ub)
                fX_std = fX_i
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

            X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                X=X_unit,
                fX=fX_std,
                length=self.length[i],
                n_training_steps=n_training_steps,
                hypers=self.hypers[i],
            )

        X_unit_next, idx_next = self._select_candidates(X_cand, y_cand)
        if n_total < self.batch_size:
            X_unit_next = X_unit_next[:n_total, :]
            idx_next = idx_next[:n_total, :]

        X_next = from_unit_cube(X_unit_next, self.lb, self.ub)
        return X_next, idx_next