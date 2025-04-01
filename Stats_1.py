# 1. Generate a random variable and display its value
random_var = np.random.rand()
print("Random Variable:", random_var)

# 2. Generate a discrete uniform distribution and plot PMF
values = np.arange(1, 7)
pmf_values = randint.pmf(values, 1, 7)
plt.bar(values, pmf_values)
plt.title("Discrete Uniform Distribution (Dice PMF)")
plt.show()

# 3. Calculate PDF of a Bernoulli distribution
def bernoulli_pdf(p):
    x = [0, 1]
    prob = bernoulli.pmf(x, p)
    return x, prob

x, prob = bernoulli_pdf(0.5)
plt.bar(x, prob)
plt.title("Bernoulli Distribution (p=0.5)")
plt.show()

# 4. Simulate a binomial distribution (n=10, p=0.5) and plot histogram
binomial_data = np.random.binomial(n=10, p=0.5, size=1000)
sns.histplot(binomial_data, discrete=True)
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.show()

# 5. Create a Poisson distribution and visualize it
poisson_data = np.random.poisson(lam=4, size=1000)
sns.histplot(poisson_data, discrete=True)
plt.title("Poisson Distribution (λ=4)")
plt.show()

# 6. Calculate and plot the CDF of a discrete uniform distribution
cdf_values = randint.cdf(values, 1, 7)
plt.plot(values, cdf_values, marker='o', linestyle='--')
plt.title("Cumulative Distribution Function (CDF)")
plt.show()

# 7. Generate a continuous uniform distribution and visualize it
uniform_data = np.random.uniform(0, 1, 1000)
sns.histplot(uniform_data, kde=True)
plt.title("Continuous Uniform Distribution")
plt.show()

# 8. Simulate data from a normal distribution and plot histogram
normal_data = np.random.normal(0, 1, 1000)
sns.histplot(normal_data, kde=True)
plt.title("Normal Distribution")
plt.show()

# 9. Calculate Z-scores and plot them
def calculate_z_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

z_scores = calculate_z_scores(normal_data)
sns.histplot(z_scores, kde=True)
plt.title("Z-Scores Distribution")
plt.show()

# 10. Implement CLT using Python for a non-normal distribution
sample_means = [np.mean(np.random.exponential(scale=1, size=30)) for _ in range(1000)]
sns.histplot(sample_means, kde=True)
plt.title("Central Limit Theorem Simulation")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from scipy import stats

# 11. Simulate multiple samples from a normal distribution and verify the Central Limit Theorem
clt_samples = [np.mean(np.random.normal(0, 1, 30)) for _ in range(1000)]
plt.hist(clt_samples, bins=30, density=True, alpha=0.6, color='g')
plt.title("Central Limit Theorem Verification")
plt.show()

# 12. Write a Python function to calculate and plot the standard normal distribution (mean = 0, std = 1)
def plot_standard_normal():
    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x, 0, 1)
    plt.plot(x, y)
    plt.title("Standard Normal Distribution (mean=0, std=1)")
    plt.show()

# 13. Generate random variables and calculate their corresponding probabilities using the binomial distribution
n, p = 10, 0.5  # Number of trials and probability of success
binomial_probs = binom.pmf(np.arange(0, 11), n, p)
plt.bar(np.arange(0, 11), binomial_probs)
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.show()

# 14. Calculate the Z-score for a given data point and compare it to a standard normal distribution
sample_value = 1.5
mean = 0  # For standard normal distribution
std_dev = 1  # For standard normal distribution
z_score = (sample_value - mean) / std_dev
print(f"Z-score for value {sample_value}: {z_score}")

# 15. Implement hypothesis testing using Z-statistics for a sample dataset
def z_test(sample, population_mean, population_std):
    z_stat = (np.mean(sample) - population_mean) / (population_std / np.sqrt(len(sample)))
    return z_stat

sample_data = np.random.normal(5, 2, 30)  # Sample with mean 5 and std 2
population_mean = 5
population_std = 2
z_statistic = z_test(sample_data, population_mean, population_std)
print(f"Z-test statistic: {z_statistic}")

# 16. Create a confidence interval for a dataset using Python and interpret the result
confidence_level = 0.95
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data)
sample_size = len(sample_data)
z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# Confidence interval calculation
margin_of_error = z_value * (sample_std / np.sqrt(sample_size))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence_level*100}% Confidence Interval: {confidence_interval}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, skew, kurtosis

# 17. Generate data from a normal distribution, then calculate and interpret the confidence interval for its mean
normal_data = np.random.normal(0, 1, 1000)  # Generate 1000 random values from a normal distribution
mean = np.mean(normal_data)
std_dev = np.std(normal_data)
n = len(normal_data)
confidence_level = 0.95

z_value = norm.ppf(1 - (1 - confidence_level) / 2)
margin_of_error = z_value * (std_dev / np.sqrt(n))

confidence_interval = (mean - margin_of_error, mean + margin_of_error)
print(f"Confidence Interval (95%): {confidence_interval}")

# 18. Calculate and visualize the probability density function (PDF) of a normal distribution
x = np.linspace(-4, 4, 100)
pdf_values = norm.pdf(x, 0, 1)
plt.plot(x, pdf_values)
plt.title("Probability Density Function (PDF) of Normal Distribution")
plt.show()

# 19. Calculate and interpret the cumulative distribution function (CDF) of a Poisson distribution
poisson_data = np.random.poisson(lam=4, size=1000)
poisson_cdf = poisson.cdf(np.arange(0, 11), 4)

plt.plot(np.arange(0, 11), poisson_cdf, marker='o')
plt.title("Cumulative Distribution Function (CDF) of Poisson Distribution (λ=4)")
plt.show()

# 20. Simulate a random variable using a continuous uniform distribution and calculate its expected value
uniform_data = np.random.uniform(0, 1, 1000)
expected_value = np.mean(uniform_data)
print(f"Expected Value of Uniform Distribution: {expected_value}")

# 21. Compare the standard deviations of two datasets and visualize the difference
dataset1 = np.random.normal(0, 1, 1000)  # Dataset with standard deviation 1
dataset2 = np.random.normal(0, 2, 1000)  # Dataset with standard deviation 2

plt.hist(dataset1, alpha=0.5, label="Std=1")
plt.hist(dataset2, alpha=0.5, label="Std=2")
plt.legend()
plt.title("Comparison of Standard Deviations")
plt.show()

# 22. Calculate the range and interquartile range (IQR) of a dataset generated from a normal distribution
data_range = np.max(normal_data) - np.min(normal_data)
iqr = np.percentile(normal_data, 75) - np.percentile(normal_data, 25)

print(f"Range: {data_range}, IQR: {iqr}")

# 23. Implement Z-score normalization on a dataset and visualize its transformation
normalized_data = (normal_data - np.mean(normal_data)) / np.std(normal_data)

plt.hist(normalized_data, bins=30, alpha=0.6, color='g')
plt.title("Z-Score Normalization of Normal Distribution Data")
plt.show()

# 24. Calculate the skewness and kurtosis of a dataset generated from a normal distribution
print(f"Skewness: {skew(normal_data)}")
print(f"Kurtosis: {kurtosis(normal_data)}")

