# Brain_Tumor_Detection
This notebook explores modular exponentiation, a fundamental operation in cryptography. It demonstrates and compares different approaches to compute base^exp mod mod.

The notebook covers the following:

#### Python Implementations:
It presents both the built-in pow() function and a custom implementation of the repeated squaring algorithm for modular exponentiation. It then benchmarks these methods to compare their performance.
#### Julia Implementation: 
It sets up a Julia runtime in Colab and implements the repeated squaring algorithm in Julia. It uses the BenchmarkTools package to measure the execution time of the custom Julia function and the built-in powermod function.
#### Further Optimization:
It demonstrates how to parallelize the modular exponentiation computation for multiple bases in Julia using multi-threading to reduce execution time.

### Result
These allows one to observe the performance difference between the Python and Julia implementations for modular exponentiation. By comparing the reported times, you would observe that for both the optimized built-in functions were highly efficient than the custom implementations. The bulit-in functions had fewer lines of code which makes it easier to write.
Generally, the Julia algorithm speed up faster than the Python, even better with multiple bases.
