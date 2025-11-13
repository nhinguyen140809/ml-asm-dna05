import torch
import re
from collections import Counter

# Choose device automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HiddenMarkovModel:
    def __init__(self, N=0, M=0, H=None, V=None, A=None, B=None, pi=None, device=DEVICE, dtype=torch.float64):
        """
        HMM implemented with PyTorch tensors.

        N: number of hidden states
        M: number of observable states
        H: list of hidden state names
        V: list of observable tokens
        A: transition matrix (N x N) as torch tensor (rows sum to 1)
        B: emission matrix (N x M)
        pi: initial distribution (N,)
        """
        self.device = device
        self.dtype = dtype

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

        self.V = V if V is not None else list(range(self.M))
        self.H = H if H is not None else list(range(self.N))

        # init tensors
        if A is not None:
            assert isinstance(A, torch.Tensor), "Transition matrix A must be a torch Tensor."
            assert A.shape == (self.N, self.N), "Transition matrix A must have shape (N, N)."
            for row in A:
                assert torch.isclose(row.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Each row of A must sum to 1."
            self.A = A.to(device=self.device, dtype=self.dtype)
        else:
            a = torch.rand((self.N, self.N), dtype=self.dtype, device=self.device)
            self.A = a / a.sum(dim=1, keepdim=True)

        if B is not None:
            assert isinstance(B, torch.Tensor), "Emission matrix B must be a torch Tensor."
            assert B.shape == (self.N, self.M), "Emission matrix B must have shape (N, M)."
            for row in B:
                assert torch.isclose(row.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Each row of B must sum to 1."
            self.B = B.to(device=self.device, dtype=self.dtype)
        else:
            b = torch.rand((self.N, self.M), dtype=self.dtype, device=self.device)
            self.B = b / b.sum(dim=1, keepdim=True)

        if pi is not None:
            assert isinstance(pi, torch.Tensor), "Initial distribution pi must be a torch Tensor."
            assert pi.shape == (self.N,), "Initial distribution pi must have shape (N,)."
            assert torch.isclose(pi.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Initial distribution pi must sum to 1."
            self.pi = pi.to(device=self.device, dtype=self.dtype)
        else:
            p = torch.rand(self.N, dtype=self.dtype, device=self.device)
            self.pi = p / p.sum()

    def __repr__(self):
        return f"HMM_Torch(N={self.N}, M={self.M}, device={self.device})\nVocab size: {self.M}\nStates: {self.N}"

    def set_V(self, V):
        assert len(V) == self.M, "Length of observable state space V must be equal to M."
        self.V = V
    
    def set_H(self, H):
        assert len(H) == self.N, "Length of hidden state space H must be equal to N."
        self.H = H

    def set_A(self, A):
        assert isinstance(A, torch.Tensor), "Transition matrix A must be a torch Tensor."
        assert A.shape == (self.N, self.N), "Transition matrix A must have shape (N, N)."
        for row in A:
            assert torch.isclose(row.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Each row of A must sum to 1."
        self.A = A

    def set_B(self, B):
        assert isinstance(B, torch.Tensor), "Emission matrix B must be a torch Tensor."
        assert B.shape == (self.N, self.M), "Emission matrix B must have shape (N, M)."
        for row in B:
            assert torch.isclose(row.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Each row of B must sum to 1."
        self.B = B

    def set_pi(self, pi):
        assert isinstance(pi, torch.Tensor), "Initial distribution pi must be a torch Tensor."
        assert pi.shape == (self.N,)
        assert torch.isclose(pi.sum(), torch.tensor(1.0, device=self.device, dtype=self.dtype)), "Initial distribution pi must sum to 1."
        self.pi = pi

    def generate_sequence(self, T):
        states, observations = [], []
        state = torch.multinomial(self.pi, 1).item()
        for _ in range(T):
            states.append(state)
            obs = torch.multinomial(self.B[state], 1).item()
            observations.append(obs)
            state = torch.multinomial(self.A[state], 1).item()
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
        alpha = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)

        # Base case: t = 0
        alpha[0] = self.pi * self.B[:, O[0]]
        
        # Recursive case: t > 0
        for t in range(1, T):
            alpha[t] = torch.matmul(alpha[t-1], self.A) * self.B[:, O[t]]

        # Termination: P(O|λ) = sum over all states of alpha[T-1, i]
        P = torch.sum(alpha[T-1]).item()
        log_P = torch.log(torch.tensor(P)) if P > 0 else float('-inf')
        
        return (log_P, alpha) if log else (P, alpha)

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
        alpha = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)
        scales = torch.zeros(T, dtype=self.dtype, device=self.device)

        # Base case: t = 0
        alpha[0] = self.pi * self.B[:, O[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]

        # Recursive case: t > 0
        for t in range(1, T):
            alpha[t] = torch.matmul(alpha[t-1], self.A) * self.B[:, O[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]

        # Compute log probability to avoid underflow
        log_P = torch.sum(torch.log(scales))
        P = torch.exp(log_P)
        log_P = log_P.item()
        P = P.item()

        return (log_P, alpha, scales) if log else (P, alpha, scales)

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
        delta = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)
        phi = torch.zeros((T, self.N), dtype=torch.int64, device=self.device)

        # Base case: t = 0
        delta[0] = self.pi * self.B[:, O[0]]
        phi[0] = 0

        # Recursive case: t > 0
        for t in range(1, T):
            values = delta[t-1].unsqueeze(1) * self.A
            max_val, max_state = torch.max(values, dim=0)
            delta[t] = max_val * self.B[:, O[t]]
            phi[t] = max_state

        # Termination: find the best path
        P_star, last_state = torch.max(delta[T-1], dim=0)

        # Backtrack to find the full path
        Q_star = torch.zeros(T, dtype=torch.int64, device=self.device)
        Q_star[T-1] = last_state
        for t in range(T-2, -1, -1):
            Q_star[t] = phi[t+1, Q_star[t+1]]

        result_states = [self.H[q.item()] for q in Q_star] if ret_tags else Q_star.tolist()
        return result_states, delta, phi

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
        beta = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)
        # Base case: t = T-1
        beta[T-1] = 1.0
        
        # Recursive case: t < T-1
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = torch.sum(self.A[i, :] * self.B[:, O[t+1]] * beta[t+1, :])

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
            
            gamma = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp(min=1e-12)


            xi = torch.zeros((T-1, self.N, self.N), dtype=self.dtype, device=self.device)
            for t in range(T-1):
                numerator = alpha[t, :].unsqueeze(1) * self.A * self.B[:, O[t+1]].unsqueeze(0) * beta[t+1, :].unsqueeze(0)
                denom = numerator.sum()
                xi[t, :, :] = numerator / denom.clamp(min=1e-12)
            
            # M-step: re-estimate A, B, pi
            self.pi = gamma[0, :] / gamma[0, :].sum().clamp(min=1e-12)
            for i in range(self.N):
                self.A[i, :] = xi[:, i, :].sum(dim=0) / xi[:, i, :].sum().clamp(min=1e-12)
                for k in range(self.M):
                    mask = (torch.tensor(O, device=self.device) == k).float()
                    self.B[i, k] = (gamma[:, i] * mask).sum() / gamma[:, i].sum().clamp(min=1e-12)

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
        print("[HMM] Starting supervised MLE training with (N, M)=({}, {})...".format(self.N, self.M))

        # Map states and observations to indices using dictionary lookup for speed
        state2idx = {s: i for i, s in enumerate(self.H)}
        obs2idx = {v: i for i, v in enumerate(self.V)}

        # Precompute mapped sequences
        mapped_states = []
        mapped_obs = []

        progress_total = len(state_sequences)
        for s_seq, o_seq in zip(state_sequences, observation_sequences):
            if len(s_seq) == 0:
                continue
            mapped_s = torch.tensor([state2idx[s] for s in s_seq], device=self.device)
            mapped_o = torch.tensor([obs2idx.get(w, obs2idx[HMMUtils.pseudo_word(w)]) for w in o_seq], device=self.device)
            mapped_states.append(mapped_s)
            mapped_obs.append(mapped_o)

        # Initialize counts
        pi_counts = torch.zeros(self.N, dtype=self.dtype, device=self.device)
        A_counts = torch.zeros((self.N, self.N), dtype=self.dtype, device=self.device)
        B_counts = torch.zeros((self.N, self.M), dtype=self.dtype, device=self.device)

        # Update counts using vectorized index_put_
        for s_seq, o_seq in zip(mapped_states, mapped_obs):
            # Initial state
            pi_counts[s_seq[0]] += 1
            # Transitions
            if len(s_seq) > 1:
                src = s_seq[:-1]
                dst = s_seq[1:]
                A_counts.index_put_((src, dst), torch.ones(len(src), device=self.device, dtype=self.dtype), accumulate=True)
            # Emissions
            B_counts.index_put_((s_seq, o_seq), torch.ones(len(s_seq), device=self.device, dtype=self.dtype), accumulate=True)

        # Normalize to probabilities (add small smoothing)
        eps = 1e-12

        self.pi = (pi_counts + eps) / (pi_counts.sum() + eps * self.N)

        denom_A = A_counts.sum(dim=1, keepdim=True).clamp(min=eps)
        self.A = (A_counts + eps) / (denom_A + eps * self.N)

        denom_B = B_counts.sum(dim=1, keepdim=True).clamp(min=eps)
        self.B = (B_counts + eps) / (denom_B + eps * self.M)

        print("[HMM] Supervised MLE training completed.")

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

    @staticmethod
    def get_pseudo_list():
        """
        Return a list of all pseudo-word classes used for rare word handling.
        """
        return [
            "<PUNCT>", "<fourDigitNum>", "<twoDigitNum>", "<othernum>",
            "<containsDigitAndAlpha>", "<containsDigitAndSlash>", "<containsDigitAndDash>",
            "<containsDigitAndComma>", "<containsDigitAndPeriod>",
            "<ALLCAPS>", "<capPeriod>", "<initCap>",
            "<suffix_ing>", "<suffix_ed>", "<suffix_ly>",
            "<lowercase>", "<other>"
        ]
    
class POS_HMM:
    """ Wrapper for creating a POS tagging HMM from training data. """
    def __init__(self, gamma=1, device=DEVICE, dtype=torch.float64):
        self.hmm = None
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

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
        
        self.hmm = HiddenMarkovModel(N=N, M=M, H=H, V=V, device=self.device, dtype=self.dtype)

        state_sequences = y_train
        observation_sequences = X_train

        # Device
        print(f"[POS_HMM] Using device: {self.device}")
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
        n = len(sentences)
        assert n > 0, "Input sentences list must be non-empty."
        result = []
        for i, sentence in enumerate(sentences):
            if i % 10 == 0 or i == n - 1:
                print(f"[POS_HMM] Predicting sentence {i+1}/{n}...", end='\r', flush=True)
            pred_tags = self.predict_sentence(sentence)
            result.append(pred_tags)
        print(f"[POS_HMM] Completed predicting {n} sentences.")
        return result
    
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

    @staticmethod
    def accuracy(true_tags, pred_tags):
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

    @staticmethod
    def precision_recall_f1(true_tags, pred_tags, average='weighted'):
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

        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        f1s = torch.tensor(f1s)
        supports = torch.tensor(supports)

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
            precision = torch.sum(precisions * supports) / torch.sum(supports) if torch.sum(supports) > 0 else 0.0
            recall = torch.sum(recalls * supports) / torch.sum(supports) if torch.sum(supports) > 0 else 0.0
            f1 = torch.sum(f1s * supports) / torch.sum(supports) if torch.sum(supports) > 0 else 0.0

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
        print("[POS_HMM] Evaluating on test data...")
        true_tags = []
        pred_tags = []
        for sentence, tags in zip(X_test, y_test):
            predicted = self.predict_sentence(sentence)
            true_tags.append(tags)
            pred_tags.append(predicted)
        
        accuracy = POS_HMM.accuracy(true_tags, pred_tags)
        precision, recall, f1 = POS_HMM.precision_recall_f1(true_tags, pred_tags, average=average)
        return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Example usage
    test_hmm = HiddenMarkovModel(N=5, M=6)
    import numpy as np
    A = torch.tensor([[0.7, 0.2, 0.05, 0.025, 0.025],
                  [0.1, 0.6, 0.2, 0.05, 0.05],
                  [0.2, 0.3, 0.4, 0.05, 0.05],
                  [0.25, 0.25, 0.25, 0.15, 0.10],
                  [0.3, 0.2, 0.2, 0.2, 0.1]], dtype=torch.float64, device=DEVICE)
    B = torch.tensor([[0.1, 0.4, 0.2, 0.2, 0.05, 0.05],
                  [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
                  [0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
                  [0.2, 0.3, 0.3, 0.1, 0.05, 0.05],
                  [0.15, 0.35, 0.25, 0.15, 0.05, 0.05]], dtype=torch.float64, device=DEVICE)
    pi = torch.tensor([0.2, 0.3, 0.25, 0.15, 0.1], dtype=torch.float64, device=DEVICE)
    test_hmm.set_A(A)
    test_hmm.set_B(B)
    test_hmm.set_pi(pi)
    # Forward algorithm test
    observations = [0, 1, 2, 2, 3, 2, 4, 2, 0, 1]
    P, alpha = test_hmm.forward(observations, is_index=True, scaled=False)
    print(f"Forward algorithm P(O|λ): {P}")
    # Viterbi algorithm test
    Q_star, delta, phi = test_hmm.Viterbi_fast(observations, is_index=True)
    print(f"Viterbi most likely states Q*: {Q_star}")
