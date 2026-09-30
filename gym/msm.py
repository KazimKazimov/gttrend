"""
Markov-switching model with covariates: a 3-state hidden Markov model whose

  * observations z_t (monthly change of the 10Y, optionally plus changes of other variables) are
    Gaussian with a state-specific mean vector and a shared (or state-specific) covariance, and
  * transition probabilities depend on lagged covariates u_{t-1} through a multinomial logit
    (time-varying transition probabilities, Diebold-Lee-Weinbach 1994, Filardo 1994):

        P(S_t = j | S_{t-1} = i, u_{t-1}) = exp(a_ij + b_ij'u_{t-1}) / sum_k exp(a_ik + b_ik'u_{t-1}),
        with a_ii = b_ii = 0 (staying is the base category).

Estimated by EM (Baum-Welch with a penalised logit M-step). Real time: filtered probabilities
P(S_t | z_1..t). Ex ante: P(S_{t+h} | z_1..t) by propagating the filter with the transition matrix at the
latest covariates. Ex post: smoothed probabilities. States are ordered by the mean 10Y change:
0 = down, 1 = sideways, 2 = up.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


class IOHMM:
    def __init__(self, K=3, l2=5.0, max_iter=150, tol=1e-6, switching_cov=False, ridge=1e-4, seed=0):
        self.K, self.l2, self.max_iter, self.tol = K, l2, max_iter, tol
        self.switching_cov, self.ridge = switching_cov, ridge
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- pieces
    def _design(self, U, T):
        if U is None:
            return np.ones((T, 1))
        return np.column_stack([np.ones(T), U])

    def _trans(self, X):
        """X: (T, p+1) design (row t = covariates at t, used for the transition into t+1).
        Returns A: (T, K, K), A[t, i, j] = P(S_{t+1}=j | S_t=i, u_t)."""
        logits = np.einsum("tp,ijp->tij", X, self.W)            # (T, K, K); W[i, i] = 0
        return np.exp(logits - logsumexp(logits, axis=2, keepdims=True))

    def _logb(self, Z):
        T, m = Z.shape
        lb = np.empty((T, self.K))
        for k in range(self.K):
            S = self.S[k] if self.switching_cov else self.S
            L = np.linalg.cholesky(S)
            r = np.linalg.solve(L, (Z - self.mu[k]).T)
            lb[:, k] = -0.5 * (r ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * m * np.log(2 * np.pi)
        return lb

    def _forward(self, lb, A):
        T = lb.shape[0]
        alpha = np.empty((T, self.K))
        logc = np.empty(T)
        a = np.log(self.pi + 1e-300) + lb[0]
        logc[0] = logsumexp(a)
        alpha[0] = np.exp(a - logc[0])
        for t in range(1, T):
            pred = alpha[t - 1] @ A[t - 1]
            a = np.log(pred + 1e-300) + lb[t]
            logc[t] = logsumexp(a)
            alpha[t] = np.exp(a - logc[t])
        return alpha, logc

    def _backward(self, lb, A, logc):
        T = lb.shape[0]
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            b = np.exp(lb[t + 1] - lb[t + 1].max())
            beta[t] = A[t] @ (b * beta[t + 1])
            beta[t] /= beta[t].sum()
        return beta

    # ---------------------------------------------------------------- fit
    def fit_supervised(self, Z, U, labels):
        """Complete-data estimation with the states fixed at `labels` (0,1,2): state means and covariance
        from the labelled months, transition logits from the labelled transitions. The regimes are then
        exactly the ex-post trend states and the forward filter does real-time inference on them."""
        Z = np.asarray(Z, float)
        T, m = Z.shape
        X = self._design(U, T)
        K = self.K
        lab = np.asarray(labels, int)
        self.mu = np.array([Z[lab == k].mean(0) if (lab == k).sum() > 1 else Z.mean(0) for k in range(K)])
        if self.switching_cov:
            self.S = np.array([np.cov(Z[lab == k].T).reshape(m, m) + self.ridge * np.eye(m) if (lab == k).sum() > m + 2
                               else np.cov(Z.T).reshape(m, m) for k in range(K)])
        else:
            R = Z - self.mu[lab]
            self.S = R.T @ R / T + self.ridge * np.eye(m)
        self.W = np.zeros((K, K, X.shape[1]))
        Y = np.zeros((T - 1, K))
        Y[np.arange(T - 1), lab[1:]] = 1
        for i in range(K):
            sel = lab[:-1] == i
            if sel.sum() < 2:
                continue
            # add one pseudo-transition to every state so that rare switches keep a finite logit
            Yi = np.vstack([Y[sel], np.eye(K)])
            Xi = np.vstack([X[:-1][sel], np.tile(X[:-1][sel].mean(0), (K, 1))])
            self.W[i] = self._fit_row(i, Yi, Xi)
        self.pi = np.bincount(lab[:12], minlength=K) + 1.0
        self.pi /= self.pi.sum()
        self.loglik_, self.n_iter_ = np.nan, 0
        return self

    def fit(self, Z, U=None, init_labels=None):
        Z = np.asarray(Z, float)
        T, m = Z.shape
        X = self._design(U, T)
        p1 = X.shape[1]
        K = self.K
        # init from labels (0,1,2) if given, else from quantiles of the first observable
        if init_labels is None:
            q = np.quantile(Z[:, 0], [1 / 3, 2 / 3])
            init_labels = np.digitize(Z[:, 0], q)
        lab = np.asarray(init_labels, int)
        self.mu = np.array([Z[lab == k].mean(0) if (lab == k).sum() > m + 1 else Z.mean(0) + (k - 1) * Z.std(0) * 0.5
                            for k in range(K)])
        resid = Z - self.mu[lab]
        S = np.cov(resid.T).reshape(m, m) + self.ridge * np.eye(m)
        self.S = np.array([S.copy() for _ in range(K)]) if self.switching_cov else S
        cnt = np.ones((K, K)) + 20 * np.eye(K)
        for a, b in zip(lab[:-1], lab[1:]):
            cnt[a, b] += 1
        P = cnt / cnt.sum(1, keepdims=True)
        self.W = np.zeros((K, K, p1))
        for i in range(K):
            for j in range(K):
                if i != j:
                    self.W[i, j, 0] = np.log(P[i, j] / P[i, i])
        self.pi = np.bincount(lab[:12], minlength=K) + 1.0
        self.pi /= self.pi.sum()

        prev = -np.inf
        for it in range(self.max_iter):
            A = self._trans(X)
            lb = self._logb(Z)
            alpha, logc = self._forward(lb, A)
            beta = self._backward(lb, A, logc)
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            # xi[t, i, j] = P(S_t=i, S_{t+1}=j | Z), t = 0..T-2
            b = np.exp(lb[1:] - lb[1:].max(1, keepdims=True))
            xi = alpha[:-1, :, None] * A[:-1] * (b * beta[1:])[:, None, :]
            xi /= xi.sum((1, 2), keepdims=True)
            ll = logc.sum()
            # M-step: emissions
            self.pi = gamma[0] + 1e-6
            self.pi /= self.pi.sum()
            w = gamma.sum(0)
            self.mu = (gamma.T @ Z) / w[:, None]
            if self.switching_cov:
                self.S = np.array([((gamma[:, k, None] * (Z - self.mu[k])).T @ (Z - self.mu[k])) / w[k]
                                   + self.ridge * np.eye(m) for k in range(K)])
            else:
                Sacc = sum(((gamma[:, k, None] * (Z - self.mu[k])).T @ (Z - self.mu[k])) for k in range(K))
                self.S = Sacc / T + self.ridge * np.eye(m)
            # M-step: transitions (penalised multinomial logit per origin state)
            for i in range(K):
                self.W[i] = self._fit_row(i, xi[:, i, :], X[:-1])
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        self.loglik_ = ll
        self.n_iter_ = it + 1
        self._order()
        return self

    def _fit_row(self, i, Y, X):
        """Y: (T-1, K) expected transition counts from state i; X: (T-1, p+1). Returns W_i (K, p+1)."""
        K, p1 = self.K, X.shape[1]
        others = [j for j in range(K) if j != i]
        n = Y.sum(1)
        pen = np.ones(p1) * self.l2
        pen[0] = 1e-3                                          # nearly free intercepts

        def unpack(theta):
            W = np.zeros((K, p1))
            W[others] = theta.reshape(len(others), p1)
            return W

        def f(theta):
            W = unpack(theta)
            lg = X @ W.T                                        # (T-1, K)
            lse = logsumexp(lg, axis=1)
            nll = -(Y * lg).sum() + (n * lse).sum()
            P = np.exp(lg - lse[:, None])
            G = -(Y - n[:, None] * P).T @ X                     # (K, p1)
            Wo = W[others]
            nll += 0.5 * (pen * Wo ** 2).sum()
            G = G[others] + pen * Wo
            return nll, G.ravel()

        theta0 = self.W[i][others].ravel()
        r = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
        return unpack(r.x)

    def _order(self):
        o = np.argsort(self.mu[:, 0])
        self.mu = self.mu[o]
        if self.switching_cov:
            self.S = self.S[o]
        self.pi = self.pi[o]
        self.W = self.W[o][:, o]
        # re-normalise so that W[i, i] = 0 (logit shift invariance per origin row)
        for i in range(self.K):
            self.W[i] = self.W[i] - self.W[i, i]

    # ---------------------------------------------------------------- inference
    def filter(self, Z, U=None):
        Z = np.asarray(Z, float)
        X = self._design(U, len(Z))
        A = self._trans(X)
        alpha, logc = self._forward(self._logb(Z), A)
        return alpha, A

    def smooth(self, Z, U=None):
        Z = np.asarray(Z, float)
        X = self._design(U, len(Z))
        A = self._trans(X)
        lb = self._logb(Z)
        alpha, logc = self._forward(lb, A)
        beta = self._backward(lb, A, logc)
        g = alpha * beta
        return g / g.sum(1, keepdims=True)

    def ahead(self, alpha, A, h):
        """P(S_{t+h} | info at t) for every t, holding covariates at their time-t values."""
        out = alpha.copy()
        for _ in range(h):
            out = np.einsum("ti,tij->tj", out, A)
        return out
