# Tango: Secure Federated Multi-Armed Bandits

DISCLAIMER: Tango is slow! We highly recommend to execute benchmarks of Tango in a server supporting intensive multi-threading. Even there, it may takes few days to complete. The reason behind this slow evaluation lies on
the homomorphic comparison using concrete. Despite its efficiency compared to other TFHE-focused libraries, FHE
remains particularly slow.

To execute Tango, **Rust** and **Cargo** are expected to be installed. 
WARNING: When compiling the tfhe library, be sure to use the nightly rust compilation toolchain, which is not installed by default.


Once installed, run the following command whose the output is located in CSV files:
```sh
cargo +nightly run --release
``` 