import numpy as np

class HiddenMarkovModel:
    def __init__(self, N=0, M=0, H=None, V=None, A=None, B=None, pi=None):
        """
        N: number of hidden states
        M: number of observable states
        V: observable state space (optional), default: {0, 1, ..., M-1}
        H: hidden state space (optional), default: {0, 1, ..., N-1}
        A: state transition probability matrix (N x N), default: random
        B: observation probability matrix (N x M), default: random
        pi: initial state distribution (N,), default: random
        """
        assert N > 0 or H is not None, "At least one of N or H must be provided."
        assert M > 0 or V is not None, "At least one of M or V must be provided."

        if N > 0:
            self.N = N
            if H is not None:
                assert len(H) == N, "Length of hidden state space H must be equal to N."
        else:
            self.N = len(H)

        if M > 0:
            self.M = M
            if V is not None:
                assert len(V) == M, "Length of observable state space V must be equal to M."
        else:
            self.M = len(V)

        self.V = V if V is not None else list(range(M))  # observable state space
        self.H = H if H is not None else list(range(N))  # hidden state space

        # Initialize randomly (normalize to probabilities)
        if A is not None:
            assert isinstance(A, np.ndarray), "Transition matrix A must be a numpy array."
            assert A.shape == (N, N), "Transition matrix A must be of shape (N, N)."
            for row in A:
                assert np.isclose(row.sum(), 1), "Each row of transition matrix A must sum to 1."
            self.A = A
        else:
            self.A = np.random.rand(N, N)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

        if B is not None:
            assert isinstance(B, np.ndarray), "Emission matrix B must be a numpy array."
            assert B.shape == (N, M), "Emission matrix B must be of shape (N, M)."
            for row in B:
                assert np.isclose(row.sum(), 1), "Each row of emission matrix B must sum to 1."
            self.B = B
        else:
            self.B = np.random.rand(N, M)
            self.B = self.B / self.B.sum(axis=1, keepdims=True)

        if pi is not None:
            assert isinstance(pi, np.ndarray), "Initial state distribution pi must be a numpy array."
            assert pi.shape == (N,), "Initial state distribution pi must be of shape (N,)."
            assert np.isclose(pi.sum(), 1), "Initial state distribution pi must sum to 1."
            self.pi = pi
        else:
            self.pi = np.random.rand(N)
            self.pi = self.pi / self.pi.sum()

    def __repr__(self):
        return f"HMM(N = {self.N}, M = {self.M})\nObservation Space V: {self.V}\nHidden Space H: {self.H}"

    def set_V(self, V):
        assert len(V) == self.M, "Length of observable state space V must be equal to M."
        self.V = V
    
    def set_H(self, H):
        assert len(H) == self.N, "Length of hidden state space H must be equal to N."
        self.H = H

    def set_A(self, A):
        assert isinstance(A, np.ndarray)
        assert A.shape == (self.N, self.N)
        for row in A:
            assert np.isclose(row.sum(), 1)
        self.A = A

    def set_B(self, B):
        assert isinstance(B, np.ndarray)
        assert B.shape == (self.N, self.M)
        for row in B:
            assert np.isclose(row.sum(), 1)
        self.B = B

    def set_pi(self, pi):
        assert isinstance(pi, np.ndarray)
        assert pi.shape == (self.N,)
        assert np.isclose(pi.sum(), 1)
        self.pi = pi

    # Có thể là index của observations trong V hoặc chính giá trị trong V
    def forward(self, observations, is_index=True):
        """
        Thuật toán Forward để tính P(O|λ).
        
        observations: list[int] - chuỗi quan sát (các chỉ số trong V)
        
        return: (P, alpha)
            - P: xác suất chuỗi quan sát
            - alpha: ma trận alpha (T x N)
        """
        if is_index:
            assert all(0 <= o < self.M for o in observations), "All observations must be valid indices in V."
            O = observations
        else:
            assert all(o in self.V for o in observations), "All observations must be in the observable state space V."
            O = [self.V.index(o) for o in observations]
        
        T = len(O)
        assert T > 0, "Observation sequence must be non-empty."

        # Initialize alpha, where alpha[t, i] = P(O[0:t], Q[t] = i | λ)
        alpha = np.zeros((T, self.N))
        # Base case: t = 0
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]
        
        # Recursive case: t > 0
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = sum(alpha[t-1, i] * self.A[i, j] for i in range(self.N)) * self.B[j, O[t]]

        # Termination: P(O|λ) = sum over all states of alpha[T-1, i]
        P = sum(alpha[T-1, i] for i in range(self.N))
        return P, alpha

# Ví dụ: 2 trạng thái ẩn, 3 loại quan sát
A=np.array([[0.7, 0.3], [0.4, 0.6]])
B=np.array([[0.1, 0.3, 0.6], [0.4, 0.2, 0.4]])
pi=np.array([0.2, 0.8])
hmm = HiddenMarkovModel(N=2, M=3, A=A, B=B, pi=pi, H=['Black', 'White'], V=['x', 'y', 'z'])
O = ['x', 'z']
P, alpha = hmm.forward(O, is_index=False)
print(f"P(O|λ) = {P:.4f}")