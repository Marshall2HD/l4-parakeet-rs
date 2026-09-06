use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/sm89/linear.cu");
    println!("cargo:rerun-if-changed=kernels/sm89/frontend.cu");
    println!("cargo:rerun-if-changed=kernels/sm89/subsampling.cu");
    println!("cargo:rerun-if-changed=kernels/sm89/encoder.cu");
    println!("cargo:rerun-if-changed=kernels/sm89/decoder.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo did not set target OS");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo did not set target arch");
    if target_os != "linux" || target_arch != "x86_64" {
        panic!("the CUDA engine is deliberately restricted to x86_64 Linux on NVIDIA L4");
    }
    println!("cargo:rustc-link-lib=dylib=cublas");

    let nvcc = env::var_os("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if !nvcc.is_file() {
        panic!(
            "nvcc was not found at {}; set NVCC to the CUDA 13 compiler",
            nvcc.display()
        );
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not set OUT_DIR"));
    let linear_cubin = out_dir.join("parakeet-sm89-linear.cubin");
    compile_cubin(
        &nvcc,
        Path::new("kernels/sm89/linear.cu"),
        &linear_cubin,
        true,
    );
    println!(
        "cargo:rustc-env=PARAKEET_SM89_CUBIN={}",
        linear_cubin.display()
    );
    let frontend_cubin = out_dir.join("parakeet-sm89-frontend.cubin");
    compile_cubin(
        &nvcc,
        Path::new("kernels/sm89/frontend.cu"),
        &frontend_cubin,
        false,
    );
    println!(
        "cargo:rustc-env=PARAKEET_SM89_FRONTEND_CUBIN={}",
        frontend_cubin.display()
    );
    let subsampling_cubin = out_dir.join("parakeet-sm89-subsampling.cubin");
    compile_cubin(
        &nvcc,
        Path::new("kernels/sm89/subsampling.cu"),
        &subsampling_cubin,
        true,
    );
    println!(
        "cargo:rustc-env=PARAKEET_SM89_SUBSAMPLING_CUBIN={}",
        subsampling_cubin.display()
    );
    let encoder_cubin = out_dir.join("parakeet-sm89-encoder.cubin");
    compile_cubin(
        &nvcc,
        Path::new("kernels/sm89/encoder.cu"),
        &encoder_cubin,
        true,
    );
    println!(
        "cargo:rustc-env=PARAKEET_SM89_ENCODER_CUBIN={}",
        encoder_cubin.display()
    );
    let decoder_cubin = out_dir.join("parakeet-sm89-decoder.cubin");
    compile_cubin(
        &nvcc,
        Path::new("kernels/sm89/decoder.cu"),
        &decoder_cubin,
        true,
    );
    println!(
        "cargo:rustc-env=PARAKEET_SM89_DECODER_CUBIN={}",
        decoder_cubin.display()
    );
}

fn compile_cubin(nvcc: &Path, source: &Path, output: &Path, fast_math: bool) {
    let mut command = Command::new(nvcc);
    command.args([
        "--cubin",
        "--std=c++17",
        "-O3",
        "-lineinfo",
        "--generate-code=arch=compute_89,code=sm_89",
        "-Xptxas=-warn-spills",
    ]);
    if fast_math {
        command.arg("--use_fast_math");
    }
    let result = command
        .arg(source)
        .arg("-o")
        .arg(output)
        .status()
        .expect("failed to invoke nvcc");

    if !result.success() {
        panic!("nvcc failed to compile {}", source.display());
    }
}
