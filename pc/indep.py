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

def main(train_path, test_path):
    X_train = load_data(train_path)
    X_test = load_data(test_path)

    probs = fit_independent_bernoulli(X_train)

    train_ll_sum, train_ll_avg = compute_log_likelihood(X_train, probs)
    test_ll_sum, test_ll_avg = compute_log_likelihood(X_test, probs)

    print(f"Train Log-Likelihood (Total): {train_ll_sum:.2f}")
    print(f"Train Log-Likelihood (Per Sample): {train_ll_avg:.4f}")
    print(f"Test Log-Likelihood (Total): {test_ll_sum:.2f}")
    print(f"Test Log-Likelihood (Per Sample): {test_ll_avg:.4f}")

main("../results/1KG/8020/data/8020_train.txt", "../results/1KG/8020/data/8020_test.txt")