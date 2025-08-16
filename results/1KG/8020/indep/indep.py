import numpy as np

def load_data(path):
    return np.loadtxt(path, dtype=int)

def fit_independent_bernoulli(X_train):
    # Estimate P(x=1) for each feature (axis=0)
    probs = X_train.mean(axis=0)
    # Avoid log(0) issues
    probs = np.clip(probs, 1e-8, 1 - 1e-8)
    return probs

def compute_log_likelihood(X, probs):
    # Bernoulli log-likelihood: x*log(p) + (1-x)*log(1-p)
    ll = X * np.log(probs) + (1 - X) * np.log(1 - probs)
    ll_per_sample = ll.sum(axis=1)  # Sum over features
    return ll_per_sample.sum(), ll_per_sample.mean()

def generate_bernoulli_samples(probs, n_samples):
    # probs: (n_features,)
    # Draw samples: shape (n_samples, n_features)
    samples = np.random.binomial(n=1, p=probs, size=(n_samples, len(probs)))
    return samples

def main(train_path, test_path, output_sample_path, n_samples=5008, n_features=10000):
    X_train = load_data(train_path)
    X_test = load_data(test_path)

    # Fit model on training data
    probs = fit_independent_bernoulli(X_train)

    # Compute log-likelihoods
    train_ll_sum, train_ll_avg = compute_log_likelihood(X_train, probs)
    test_ll_sum, test_ll_avg = compute_log_likelihood(X_test, probs)

    print(f"Train Log-Likelihood (Total): {train_ll_sum:.2f}")
    print(f"Train Log-Likelihood (Per Sample): {train_ll_avg:.4f}")
    print(f"Test Log-Likelihood (Total): {test_ll_sum:.2f}")
    print(f"Test Log-Likelihood (Per Sample): {test_ll_avg:.4f}")

    samples = generate_bernoulli_samples(probs, n_samples)
    np.savetxt(output_sample_path, samples, fmt="%d")

    print(f"Generated samples of shape {samples.shape} saved to {output_sample_path}")

main(
    train_path="/scratch2/prateek/genetic_pc/reproduce_final/8020/data/8020_train.txt",
    test_path="/scratch2/prateek/genetic_pc/reproduce_final/8020/data/8020_test.txt",
    output_sample_path="/scratch2/prateek/genetic_pc/reproduce_final/8020/indep/10K_indep_8020_samples.txt",
    n_samples=5008,
    n_features=10000
)
