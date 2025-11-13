import numpy as np
import re
from collections import Counter

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

    def generate_sequence(self, T):
        states, observations = [], []
        state = np.random.choice(self.N, p=self.pi)
        for _ in range(T):
            obs = np.random.choice(self.M, p=self.B[state])
            states.append(state)
            observations.append(obs)
            state = np.random.choice(self.N, p=self.A[state])
        return [self.H[s] for s in states], [self.V[o] for o in observations]
    
    def check_observations(self, observations, is_index=True):
        if is_index:
            assert all(0 <= o < self.M for o in observations), "All observations must be valid indices in V."
            O = observations
        else:
            assert all(o in self.V for o in observations), "All observations must be in the observable state space V."
            O = [self.V.index(o) for o in observations]
        return O

    def forward(self, observations, is_index=True, scaled=False, log=False):
        """
        Forward algorithm to compute P(O|λ).

        observations: list[int] - observation sequence (indices in V) or list[var] - observation sequence (elements in V)
        is_index: bool - True if observations are indices, False if they are elements in V
        scaled: bool - True to use scaled version to avoid underflow, False for unscaled version

        return: (P, alpha) or (P, alpha, scales)
            - P: probability of the observation sequence given the model λ
            - alpha: alpha matrix (T x N)
            - scales: scaling factors for each time step (T,) (only if scaled=True)
        """
        if scaled:
            return self.forward_scaled(observations, is_index, log)
        else:
            return self.forward_unscaled(observations, is_index, log)

    def forward_unscaled(self, observations, is_index=True, log=False):
        """
        Forward algorithm to compute P(O|λ).

        observations: list[int] - observation sequence (indices in V) or list[var] - observation sequence (elements in V)
        is_index: bool - True if observations are indices, False if they are elements in V

        return: (P, alpha)
            - P: probability of the observation sequence given the model λ
            - alpha: alpha matrix (T x N)
        """
        O = self.check_observations(observations, is_index)
        
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
        log_P = np.log(P) if P > 0 else float('-inf')
        if log:
            return log_P, alpha
        else:
            return P, alpha

    def forward_scaled(self, observations, is_index=True, log=False):
        """
        Scaled Forward algorithm to compute P(O|λ) and avoid underflow.

        observations: list[int] - observation sequence (indices in V) or list[var] - observation sequence (elements in V)
        is_index: bool - True if observations are indices, False if they are elements in V

        return: (P, alpha, scales)
            - P: probability of the observation sequence given the model λ
            - alpha: scaled alpha matrix (T x N)
            - scales: scaling factors for each time step (T,)

        Reasoning: To avoid numerical underflow when dealing with long sequences, we scale the alpha values at each time step. The scaling factors are stored in the 'scales' array, and the final probability is computed using these scaling factors.
        """
        O = self.check_observations(observations, is_index)
        
        T = len(O)
        assert T > 0, "Observation sequence must be non-empty."

        # Initialize alpha and scales
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        # Base case: t = 0
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]

        # Recursive case: t > 0
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = sum(alpha[t-1, i] * self.A[i, j] for i in range(self.N)) * self.B[j, O[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]

        # Compute log probability to avoid underflow
        log_P = np.sum(np.log(scales))
        P = np.exp(log_P)
        if log:
            return log_P, alpha, scales
        else:
            return P, alpha, scales

    def Viterbi(self, observations, is_index=True, ret_tags=False):
        """
        Viterbi algorithm to find the most likely hidden state sequence Q*.
        observations: list[int] - observation sequence (indices in V)
        return: (Q_star, delta, phi)
            - Q_star: most likely hidden state sequence
            - delta: delta matrix (T x N)
            - phi: phi matrix (T x N)
        """
        O = self.check_observations(observations, is_index)
        
        T = len(O)
        assert T > 0, "Observation sequence must be non-empty."

        # Initialize delta and phi
        delta = np.zeros((T, self.N))
        phi = np.zeros((T, self.N), dtype=int)

        # Base case: t = 0
        for i in range(self.N):
            delta[0, i] = self.pi[i] * self.B[i, O[0]]
            phi[0, i] = 0

        # Recursive case: t > 0
        for t in range(1, T):
            for j in range(self.N):
                max_val, max_state = max((delta[t-1, i] * self.A[i, j], i) for i in range(self.N))
                delta[t, j] = max_val * self.B[j, O[t]]
                phi[t, j] = max_state

        # Termination: find the best path
        P_star = max(delta[T-1, i] for i in range(self.N))
        last_state = np.argmax(delta[T-1, :])

        # Backtrack to find the full path
        Q_star = [0] * T
        Q_star[T-1] = last_state
        for t in range(T-2, -1, -1):
            Q_star[t] = phi[t+1, Q_star[t+1]]

        if ret_tags:
            return [self.H[i] for i in Q_star], delta, phi
        return Q_star, delta, phi

    def backward(self, observations, is_index=True):
        """
        Backward algorithm to compute P(O|λ).

        observations: list[int] - observation sequence (indices in V) or list[var] - observation sequence (elements in V)
        is_index: bool - True if observations are indices, False if they are elements in V
        scaled: bool - True to use scaled version to avoid underflow, False for unscaled version

        return: beta matrix (T x N)
        """
        if is_index:
            assert all(0 <= o < self.M for o in observations), "All observations must be valid indices in V."
            O = observations
        else:
            assert all(o in self.V for o in observations), "All observations must be in the observable state space V."
            O = [self.V.index(o) for o in observations]
        
        T = len(O)
        assert T > 0, "Observation sequence must be non-empty."

        # Initialize beta, where beta[t, i] = P(O[t+1:T] | Q[t] = i, λ)
        beta = np.zeros((T, self.N))
        # Base case: t = T-1
        for i in range(self.N):
            beta[T-1, i] = 1
        
        # Recursive case: t < T-1
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = sum(self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j] for j in range(self.N))

        return beta
    
    def BaumWelch(self, observations, is_index=True, max_iter=100, tol=1e-6):
        """
        Baum-Welch algorithm to train the HMM parameters using the given observation sequence.

        observations: list[int] - observation sequence (indices in V) or list[var] - observation sequence (elements in V)
        is_index: bool - True if observations are indices, False if they are elements in V
        max_iter: int - maximum number of iterations
        tol: float - tolerance for convergence

        return: log likelihood of the observation sequence given the model λ
        """
        
        if is_index:
            assert all(0 <= o < self.M for o in observations), "All observations must be valid indices in V."
            O = observations
        else:
            assert all(o in self.V for o in observations), "All observations must be in the observable state space V."
            O = [self.V.index(o) for o in observations]
        
        T = len(O)
        assert T > 0, "Observation sequence must be non-empty."

        for _ in range(max_iter):
            # E-step: compute alpha, beta, gamma, xi
            _, alpha = self.forward(O, is_index=True, scaled=False)
            beta = self.backward(O, is_index=True)
            
            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T):
                denom = sum(alpha[t, i] * beta[t, i] for i in range(self.N))
                for i in range(self.N):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / denom if denom > 0 else 0

            for t in range(T-1):
                denom = sum(alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j] for i in range(self.N) for j in range(self.N))
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]) / denom if denom > 0 else 0
            
            # M-step: re-estimate A, B, pi
            for i in range(self.N):
                self.pi[i] = gamma[0, i]
                for j in range(self.N):
                    numer = sum(xi[t, i, j] for t in range(T-1))
                    denom = sum(gamma[t, i] for t in range(T-1))
                    self.A[i, j] = numer / denom if denom > 0 else 0

                for k in range(self.M):
                    numer = sum(gamma[t, i] for t in range(T) if O[t] == k)
                    denom = sum(gamma[t, i] for t in range(T))
                    self.B[i, k] = numer / denom if denom > 0 else 0

    def train_supervised_MLE(self, state_sequences, observation_sequences):
        """
        Supervised MLE training for HMM using counts from labeled sequences.

        state_sequences: list of lists of hidden states (tags)
        observation_sequences: list of lists of observed tokens (words)
        gamma: int, cutoff for rare words → pseudo-word (optional)
        word_counts: Counter, frequency of words in training (required if gamma is set)

        After this, self.pi, self.A, self.B are updated.
        """
        assert len(state_sequences) == len(observation_sequences), "Mismatch in number of sequences between states and observations."

        # Map observations to indices (apply pseudo-word mapping)
        mapped_sequences = []
        for obs_seq in observation_sequences:
            mapped_seq = []
            for w in obs_seq:
                if w not in self.V:
                    w = HMMUtils.pseudo_word(w)
            mapped_sequences.append(mapped_seq)

        # Map states to indices
        state_indices_sequences = []
        for seq in state_sequences:
            idx_seq = [self.H.index(s) for s in seq]
            state_indices_sequences.append(idx_seq)

        # Initialize counts
        N = self.N
        M = self.M
        pi_counts = np.zeros(N, dtype=float)
        A_counts = np.zeros((N, N), dtype=float)
        B_counts = np.zeros((N, M), dtype=float)

        for s_seq, o_seq in zip(state_indices_sequences, mapped_sequences):
            if len(s_seq) == 0:
                continue
            # initial state
            pi_counts[s_seq[0]] += 1
            for t in range(len(s_seq)):
                B_counts[s_seq[t], o_seq[t]] += 1
                if t < len(s_seq)-1:
                    A_counts[s_seq[t], s_seq[t+1]] += 1

        # Normalize to probabilities (add small smoothing)
        eps = 1e-12
        self.pi = (pi_counts + eps) / (pi_counts.sum() + eps * N)

        self.A = np.zeros_like(A_counts)
        for i in range(N):
            denom = A_counts[i].sum()
            if denom > 0:
                self.A[i] = (A_counts[i] + eps) / (denom + eps * N)
            else:
                self.A[i] = np.ones(N) / N

        self.B = np.zeros_like(B_counts)
        for i in range(N):
            denom = B_counts[i].sum()
            if denom > 0:
                self.B[i] = (B_counts[i] + eps) / (denom + eps * M)
            else:
                self.B[i] = np.ones(M) / M


class HMMUtils:

    # Compile regex patterns for efficiency
    DIGIT_4_RE = re.compile(r'^\d{4}$')
    DIGIT_2_RE = re.compile(r'^\d{2}$')
    DIGIT_RE = re.compile(r'^\d+$')
    DIGIT_ALPHA_RE = re.compile(r'(?=.*\d)(?=.*[A-Za-z])') 
    SLASH_RE = re.compile(r'\d+/\d+(/\d+)?') # e.g., 12/31 or 12/31/2023
    DASH_RE = re.compile(r'\d+-\d+')         # e.g., 2023-01
    COMMA_RE = re.compile(r'\d+,\d+')       # e.g., 1,000
    PERIOD_RE = re.compile(r'\d+\.\d+')      # e.g., 1.23
    CAP_PERIOD_RE = re.compile(r'^[A-Z]\.$') # M.

    @staticmethod
    def pseudo_word(token: str) -> str:
        """
        Map a raw token (string, original casing) to a pseudo-word class.
        Rules inspired by Jurafsky & Martin lecture notes (initCap, fourDigitNum, ...).

        Order of checks matters: more specific patterns should be checked first.

        Args:
            token (str): raw token (non-empty string)
        
        Returns:
            str: pseudo-word class corresponding to the token
        """
        assert token and isinstance(token, str), "Token must be a non-empty string."
        t = token
        t_lower = t.lower()  # for suffix checks

        # Punctuation / special characters
        if all(ch in ".,;:!?\"'()[]{}" for ch in t):
            return "<PUNCT>"

        # Number forms / containing numbers
        if HMMUtils.SLASH_RE.search(t):
            return "<containsDigitAndSlash>"
        if HMMUtils.DASH_RE.search(t):
            return "<containsDigitAndDash>"
        if HMMUtils.COMMA_RE.search(t):
            return "<containsDigitAndComma>"
        if HMMUtils.PERIOD_RE.search(t):
            return "<containsDigitAndPeriod>"
        if HMMUtils.DIGIT_ALPHA_RE.search(t):
            return "<containsDigitAndAlpha>"
        if HMMUtils.DIGIT_4_RE.fullmatch(t):
            return "<fourDigitNum>"
        if HMMUtils.DIGIT_2_RE.fullmatch(t):
            return "<twoDigitNum>"
        if HMMUtils.DIGIT_RE.fullmatch(t):
            return "<othernum>"

        # Capitalized / uppercase patterns
        if t.isupper():
            return "<ALLCAPS>"  
        if HMMUtils.CAP_PERIOD_RE.fullmatch(t):
            return "<capPeriod>"
        if t[0].isupper() and t[1:].islower():
            return "<initCap>"

        # Suffix heuristics
        if len(t) >= 4 and t_lower.endswith("ing"):
            return "<suffix_ing>"
        if len(t) >= 3 and t_lower.endswith("ed"):
            return "<suffix_ed>"
        if len(t) >= 3 and t_lower.endswith("ly"):
            return "<suffix_ly>"

        # Lowercase words
        if t.islower():
            return "<lowercase>"

        # Fallback
        return "<other>"


class POS_HMM:
    """ Wrapper for creating a POS tagging HMM from training data. """
    def __init__(self, gamma=1):
        self.hmm = None
        self.gamma = gamma

    def train(self, X_train, y_train, tagset=None):
        """
        Train the POS tagging HMM from training data.
        train_data: list of (sentence, tags) pairs
            - sentence: list of words (tokens)
            - tags: list of corresponding POS tags
        tagset: list or set of all possible tags (optional)
        """
        assert all(len(s) == len(t) for s, t in zip(X_train, y_train)), "Each sentence and tag sequence must be of the same length."
        
        # Collect unique tags and words
        if tagset is not None:
            assert isinstance(tagset, (list, set)), "tagset must be a list or set of tags."
            all_tags = set(tagset)
        else:
            all_tags = set()
            for tags in y_train:
                all_tags.update(tags)
        assert len(all_tags) > 0, "No tags found in training data."

        # Build vocabulary with pseudo-words for rare words
        word_counts = Counter()
        for sentence in X_train:
            word_counts.update(sentence)

        vocab = set()
        for word, count in word_counts.items():
            if count >= self.gamma:
                vocab.add(word)

        vocab.update(HMMUtils.get_pseudo_list())

        V = sorted(vocab)
        H = sorted(all_tags)
        N = len(H)
        M = len(V)

        print(f"[POS_HMM] Start training HMM for POS tagging...")
        
        self.hmm = HiddenMarkovModel(N=N, M=M, H=H, V=V)

        state_sequences = y_train
        observation_sequences = X_train

        self.hmm.train_supervised_MLE(
            state_sequences=state_sequences,
            observation_sequences=observation_sequences
        )

        print("[POS_HMM] Training complete successfully!")

    def __repr__(self):
        return f"POS_HMM with {self.hmm.N} states and {self.hmm.M} observations."

    def predict_sentence(self, sentence):
        """
        Predict POS tags for a given sentence.
        sentence: list of words (tokens)
        return: list of predicted tags
        """
        assert self.hmm is not None, "HMM model is not trained yet."

        mapped_sentence = []
        for w in sentence:
            mapped_sentence.append(HMMUtils.pseudo_word(w) if w not in self.hmm.V else w)

        predicted_tags, _, _ = self.hmm.Viterbi(mapped_sentence, is_index=False, ret_tags=True)

        return predicted_tags
    
    def predict_batch(self, sentences):
        """
        Predict POS tags for a batch of sentences.
        sentences: list of sentences, where each sentence is a list of words (tokens)
        return: list of lists of predicted tags
        """
        return [self.predict_sentence(sentence) for sentence in sentences]
    
    def predict(self, X):
        """
        Predict POS tags for input data X.
        X: list of sentences, where each sentence is a list of words (tokens)
        return: list of lists of predicted tags
        """
        assert self.hmm is not None, "HMM model is not trained yet."
        assert isinstance(X, list), "Input X must be a list of sentences."

        if all(isinstance(s, list) for s in X):
            print("[POS_HMM] Predicting in batch mode...")
            return self.predict_batch(X)
        else:
            print("[POS_HMM] Predicting single sentence...")
            return self.predict_sentence(X)

    def accuracy(self, true_tags, pred_tags):
        """ 
        Compute accuracy given true and predicted tags
        true_tags: list of true tags of all samples
        pred_tags: list of predicted tags of all samples
        return: float, accuracy score
        """
        assert len(true_tags) == len(pred_tags), "Length of true_tags and pred_tags must be the same."
        for t, p in zip(true_tags, pred_tags):
            assert len(t) == len(p), "Each pair of true and predicted tag sequences must be of the same length."
        correct = sum(ti == pi for t, p in zip(true_tags, pred_tags) for ti, pi in zip(t, p))
        total = sum(len(t) for t in true_tags)

        return correct / total if total > 0 else 0.0

    def precision_recall_f1(self, true_tags, pred_tags, average='weighted'):
        """
        Compute precision, recall, and F1 for multi-class tagging.
        Supports 'micro', 'macro', and 'weighted' averaging.

        Args:
            true_tags: list of list of true tags
            pred_tags: list of list of predicted tags
            average: str, one of ['micro', 'macro', 'weighted']

        Returns:
            precision, recall, f1 (floats)
        """
        # Flatten all sequences
        y_true = [ti for t in true_tags for ti in t]
        y_pred = [pi for p in pred_tags for pi in p]

        # Count true positives, false positives, false negatives per class
        labels = sorted(set(y_true + y_pred))
        tp = Counter()
        fp = Counter()
        fn = Counter()
        support = Counter()

        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                tp[yt] += 1
            else:
                fp[yp] += 1
                fn[yt] += 1
            support[yt] += 1

        # Compute per-class precision, recall, f1
        precisions, recalls, f1s, supports = [], [], [], []
        for label in labels:
            p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            supports.append(support[label])

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        supports = np.array(supports)
        total_support = supports.sum()

        # Average modes
        if average == 'micro':
            total_tp = sum(tp.values())
            total_fp = sum(fp.values())
            total_fn = sum(fn.values())
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        elif average == 'macro':
            precision = precisions.mean()
            recall = recalls.mean()
            f1 = f1s.mean()

        elif average == 'weighted':
            precision = np.average(precisions, weights=supports)
            recall = np.average(recalls, weights=supports)
            f1 = np.average(f1s, weights=supports)

        else:
            raise ValueError("average must be one of ['micro', 'macro', 'weighted']")

        return precision, recall, f1

    def evaluate(self, X_test, y_test, average='weighted'):
        """
        Evaluate the HMM POS tagger on test data.
        test_data: list of (sentence, true_tags) pairs
        return: float, accuracy score
        """
        assert self.hmm is not None, "HMM model is not trained yet."
        true_tags = []
        pred_tags = []
        for sentence, tags in zip(X_test, y_test):
            predicted = self.predict_sentence(sentence)
            true_tags.append(tags)
            pred_tags.append(predicted)
        
        accuracy = self.accuracy(true_tags, pred_tags)
        precision, recall, f1 = self.precision_recall_f1(true_tags, pred_tags, average=average)
        return accuracy, precision, recall, f1