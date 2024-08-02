import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import math
path = "Problem7Data.xls"
data = pd.read_excel(path)
print(data.head())
columns = data.columns
fig_count = 0
for i,col in enumerate(columns):
    plt.figure(fig_count)
    sns.histplot(data[col], bins=10, alpha=0.5, kde=True, stat="density")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'histogram for {col}')
    fig_count += 1

rate = (data[columns[1]]/data[columns[0]])*100
plt.figure(fig_count)
sns.histplot(rate, bins=10, alpha=0.5, kde=True, stat="density")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title(f'histogram for mortalitiy rate')
fig_count += 1

#part b#
print("\n part b \n")
pop_mean = data[columns[0]].mean()
pop_var = data[columns[0]].var()
pop_std = data[columns[0]].std()
total_cancer = data[columns[1]].sum()
cancer_mean = data[columns[1]].mean()
cancer_var = data[columns[1]].var()
cancer_std = data[columns[1]].std()
print(f'population mean is : {pop_mean}')
print(f'variance of population : {pop_var}')
print(f'standard devation : {pop_std}')
print(f'total cancer mortality : {total_cancer}')
print(f'cancer mortality mean : {cancer_mean}')
print(f'cancer mortality std : {cancer_std}')
print(f'cancer mortality var : {cancer_var}')


#part c#
print("\n part c \n")

simulation = 1000
sample_size = 25

def getting_sample(data_lenth, sample_size):
    sample = np.zeros(shape=sample_size) 
    for i in range(sample_size):
        sample[i] = np.random.randint(0, data_lenth)
    return sample

def simulate_samples(data, number_of_simulations, sample_size, columns):
    means_vector = np.zeros(shape=number_of_simulations)
    for i in range(number_of_simulations):
        sample = getting_sample(len(data), sample_size)
        means_vector[i] = data[columns[1]][sample].mean()
    return means_vector

means_vec = simulate_samples(data, simulation, sample_size, columns)

plt.figure(fig_count)
fig_count += 1
sns.histplot(means_vec, bins=20, alpha=0.7, kde=True, stat="density")
plt.title('histogram of 1000 simulation means')
plt.xlabel('sample')
plt.ylabel('mean')

# part d#
print("\n part d \n")
sample = getting_sample(len(data), sample_size)

estimated_mean_cancer = data[columns[1]][sample].mean()
estimated_total_cancer = estimated_mean_cancer * len(data)
estimated_var_caner = (sample_size / (sample_size - 1))* data[columns[1]][sample].var()
estimated_std_cancer = math.sqrt(estimated_var_caner)
print(f'esimated mean cancer for a sample : {estimated_mean_cancer}') 
print(f'estimated total cancer for a sample : {estimated_total_cancer}')

#part e#
print("\n part e \n")
estimated_pop_mean = data[columns[0]][sample].mean()
estimated_pop_var = data[columns[0]][sample].var() * (sample_size/(sample_size -1))
estimated_pop_std = math.sqrt(estimated_pop_var)
print(f'estimated population variance is: {estimated_pop_var}')
print(f'estimated population standard deviation is : {estimated_pop_std}')

#part f#
print("\n part f \n")

degree_freedom = sample_size-1
alpha = 0.025
score = 1-alpha
ppf = t.ppf(score, degree_freedom)
m_error_pop = ppf * (estimated_pop_std/math.sqrt(sample_size))
m_error_cancer = ppf * (estimated_std_cancer/math.sqrt(sample_size))
m_error_total = m_error_cancer * len(data)
print(f'CI for mean of population for sample: {estimated_pop_mean - m_error_pop, estimated_pop_mean + m_error_pop}')
print(f'CI for mean of cancer for sample : {estimated_mean_cancer - m_error_cancer, estimated_mean_cancer + m_error_cancer}')
print(f'CI for total cancer for sample : {estimated_total_cancer - m_error_total, estimated_total_cancer + m_error_total}')

#part g#
print("\n part g \n")
sample_size = 100
sample = getting_sample(len(data), sample_size)


estimated_mean_cancer = data[columns[1]][sample].mean()
estimated_total_cancer = estimated_mean_cancer * len(data)
estimated_var_caner = (sample_size / (sample_size - 1))* data[columns[1]][sample].var()
estimated_std_cancer = math.sqrt(estimated_var_caner)
print(f'esimated mean cancer for a sample : {estimated_mean_cancer}') 
print(f'estimated total cancer for a sample : {estimated_total_cancer}')


estimated_pop_mean = data[columns[0]][sample].mean()
estimated_pop_var = data[columns[0]][sample].var() * (sample_size/(sample_size -1))
estimated_pop_std = math.sqrt(estimated_pop_var)
print(f'estimated population variance is: {estimated_pop_var}')
print(f'estimated population standard deviation is : {estimated_pop_std}')


m_error_pop = 1.96 * (estimated_pop_std/math.sqrt(sample_size))
m_error_cancer = 1.96 * (estimated_std_cancer/math.sqrt(sample_size))
m_error_total = m_error_cancer * len(data)
print(f'CI for mean of population for sample: {estimated_pop_mean - m_error_pop, estimated_pop_mean + m_error_pop}')
print(f'CI for mean of cancer for sample : {estimated_mean_cancer - m_error_cancer, estimated_mean_cancer + m_error_cancer}')
print(f'CI for total cancer for sample : {estimated_total_cancer - m_error_total, estimated_total_cancer + m_error_total}')

# part h#
print("\n part h \n")

plt.figure(fig_count)
fig_count+=1
sns.scatterplot(x=columns[0], y=columns[1], data=data)
plt.xlabel('population')
plt.ylabel('mortalities')
plt.title('correlation')

# part i#
print("\n part i \n")

def simulate_ratio_sampels(data, number_of_simulations, sample_size, columns):
    means = np.zeros(number_of_simulations)
    for i in range(number_of_simulations):
        sample = getting_sample(len(data), sample_size)
        means[i] = ((data[columns[1]][sample].mean())/(data[columns[0]][sample].mean()))*pop_mean
    return means

sample_size = 25
means = simulate_ratio_sampels(data, simulation, sample_size, columns)
plt.figure(fig_count)
fig_count+=1
sns.histplot(means, stat='density', kde=True, bins=20, alpha=0.5)
plt.title('1000 simulation with ratio for mean of mortality')
plt.xlabel('smaple')
plt.ylabel('mean with ratio')

# part j#
print("\n part j \n")

sample = getting_sample(len(data), sample_size)
estimated_mean_cancer = data[columns[1]][sample].mean()
estimated_total_cancer = estimated_mean_cancer * len(data)
estimated_var_caner = (sample_size / (sample_size - 1))* data[columns[1]][sample].var()
estimated_std_cancer = math.sqrt(estimated_var_caner)
print(f'esimated mean cancer for a sample : {estimated_mean_cancer}') 
print(f'estimated total cancer for a sample : {estimated_total_cancer}')

estimated_mean_cancer_ratio = pop_mean * (data[columns[1]][sample].mean()/data[columns[0]][sample].mean())
estimated_total_cancer_ratio= estimated_mean_cancer_ratio * len(data)
print(f'esimated mean ratio cancer for sampe sample : {estimated_mean_cancer_ratio}')
print(f'estimated total cancer ratio for sampe sample : {estimated_total_cancer_ratio}')

#part k#
print("\n part k \n")

X_bar = data[columns[0]][sample].mean()
Y_bar = data[columns[1]][sample].mean()
Ratio = Y_bar/X_bar
var_x = data[columns[0]][sample].var()
var_x = var_x * sample_size /(sample_size-1)
var_y = data[columns[1]][sample].var()
var_y = var_y * sample_size / (sample_size-1)
sxy = np.array((data[columns[0]][sample]-X_bar))*np.array((data[columns[1]][sample]-Y_bar))
sxy = (1/(sample_size-1))*sxy.sum()
Sr = (1/sample_size)*(Ratio**2 * var_x + var_y - 2*Ratio*sxy)/(X_bar**2)
Sr = math.sqrt(Sr)
print(f'CI for mean cancer using ratio estimator: {(Ratio - ppf*Sr)*pop_mean, (Ratio + ppf*Sr)*pop_mean}')
print(f'CI for total cancer mortality using ratio estimator: {(Ratio - ppf*Sr)*pop_mean*len(data), (Ratio + ppf*Sr)*pop_mean*len(data)}')
Ratio = X_bar / Y_bar
Sr = (1/sample_size)*(Ratio**2 * var_y + var_x - 2*Ratio*sxy)/(Y_bar**2)
Sr = math.sqrt(Sr)
print(f'CI for mean population using ratio estimator: {(Ratio - ppf*Sr)*cancer_mean, (Ratio + ppf*Sr)*cancer_mean}')

#part l#
print("\n part l \n")
strata = 4
def make_stratums(data, strata, columns):
    stratums = [0 for i in range(strata)]
    precentiles = 0
    q = np.zeros(shape=strata)
    for i in range(strata):
        precentiles += (1/strata)
        q[i] = data[columns[0]].quantile(precentiles)
        if(i == 0):
            stratums[i] = data[data[columns[0]] <= q[0]]
        else:
            stratums[i] = data[(data[columns[0]] > q[i-1]) & (data[columns[0]] <= q[i])]
    return stratums

def sample_each_strata(stratums, sample_size = 6):
    return stratums.sample(sample_size, replace=True)

def whole_sample(stratums, sample_strata):
    sample = [0 for _ in range(len(stratums))]
    for i in range(len(stratums)):
        sample[i] = sample_each_strata(stratums[i], sample_strata[i])
    return sample

def mean_each_strata(samples, columns):
    means = [0 for _ in range(len(samples))]
    for i in range(len(samples)):
        means[i] = samples[i][columns].mean()
    return means

def estimate_mean_strata(stratums, mean_of_stratas):
    sum = 0
    mean = 0
    for i in range(len(stratums)):
        sum += len(stratums[i])
    for i in range(len(stratums)):
        mean += (len(stratums[i])/sum)*mean_of_stratas[i]
    return mean


stratums = make_stratums(data, strata, columns)
# for i in range(strata):
#     print(f'{i+1}th strata')
#     print(stratums[i][columns[0]].mean())
#     print(stratums[i][columns[0]].std())
#     print(stratums[i][columns[1]].mean())
#     print(stratums[i][columns[1]].std())
#     print(len(stratums[i]))
#     print("\n")

sample_per_strata = [6 for _ in range(strata)]
samples = whole_sample(stratums, sample_per_strata)
cancer_means_strata = mean_each_strata(samples, columns[1])
pop_means_strata = mean_each_strata(samples, columns[0])
estimated_mean_cancer =  estimate_mean_strata(stratums, cancer_means_strata)
estimated_mean_pop = estimate_mean_strata(stratums, pop_means_strata)
print(f'Estimate mean cancer is : {estimated_mean_cancer}')
print(f'Estimate population mean is : {estimated_mean_pop}')
print(f'Estimate total cancer mortality is : {len(data) * estimated_mean_cancer}')

means_of_cancer = np.zeros(shape=simulation)
for i in range(simulation):
    samples = whole_sample(stratums, sample_per_strata)
    cancer_means_strata = mean_each_strata(samples, columns[1])
    means_of_cancer[i] = estimate_mean_strata(stratums, cancer_means_strata)

plt.figure(fig_count)
fig_count+=1
sns.histplot(means_of_cancer, bins=20, alpha=0.5, kde=True, stat='density')
plt.xlabel('sample')
plt.ylabel('mean of sample')
plt.title('1000 simulation for stratify')
# plt.show()

#part m#
print("\n part m \n")
def optimal_sample(stratums, column, sample_size=24):
    weighted_sum_std = 0
    weighted_sum_mean = 0
    opt_sample = np.zeros(shape=len(stratums))
    for i in range(len(stratums)):
        weighted_sum_std += (len(stratums[i])*stratums[i][column].std())
        weighted_sum_mean += (len(stratums[i])*stratums[i][column].mean())
    for i in range(len(stratums)):
        opt_sample[i] = (len(stratums[i])*stratums[i][column].std())/weighted_sum_std
    return np.round(opt_sample*sample_size), weighted_sum_std, weighted_sum_mean

def compare_opt_prop(stratums, column, simga_hat, data_len):
    weights = np.zeros(len(stratums))
    compar = 0
    for i in range(len(stratums)):
        weights[i] = len(stratums[i])/data_len
        compar += (weights[i] * ((stratums[i][column].std()-simga_hat)**2))
    return compar

def compare_simple_proportional(stratums, column, mu, data_len):
    weights = np.zeros(len(stratums))
    compar = 0
    for i in range(len(stratums)):
        weights[i] = len(stratums[i])/data_len
        compar += (weights[i] * ((stratums[i][column].mean()-mu)**2))
    return compar
opt_samples, weighted_std, weighted_mean = optimal_sample(stratums, columns[1], sample_size=24)
weighted_std = weighted_std/len(data)
weighted_mean = weighted_mean/len(data)
print(opt_samples)

compare1 = compare_opt_prop(stratums, columns[1], weighted_std, len(data))
print(f'compare between proportional and optimal allocation is : {compare1/24}')

compare2 = compare_simple_proportional(stratums, columns[1], weighted_mean, len(data))
print(f'compare between simple random sample and proportional allocation is: {compare2/24}')
print(f'compare between simple random sample and optimal allocation is: {(compare1+compare2)/24}')

#part n#
print("\n part n \n")
stratas = [8, 16, 32, 64]

for i in range(len(stratas)):
    print(f'For strata = {stratas[i]}')
    stratums = make_stratums(data, stratas[i], columns)
    _, weighted_std, weighted_mean = optimal_sample(stratums, columns[1])
    weighted_mean = weighted_mean/len(data)
    weighted_std = weighted_std/len(data)
    compare1 = compare_opt_prop(stratums, columns[1], weighted_std, len(data))
    compare2 = compare_simple_proportional(stratums, columns[1], weighted_mean, len(data))
    print(f'compare between proportional and optimal allocation is : {compare1/24}')
    print(f'compare between simple random sample and proportional allocation is: {compare2/24}')
    print(f'compare between simple random sample and optimal allocation is: {(compare1+compare2)/24}\n')

