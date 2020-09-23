# pyext-rustlib

This is an optimized implementation of the index provider in Rust. It builds as a [pyo3](https://github.com/PyO3/pyo3) extension that can be called from Python.

The pre-packaged library file (`../rustlib.so`) works on x86-64 GNU/Linux.

## How to Build

1. Install Nightly build of Rust: `curl https://sh.rustup.rs -sSf | sh`
2. Run `./build.sh`
