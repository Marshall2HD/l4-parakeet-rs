use cudarc::driver::{CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut};
use half::f16;
use std::error::Error;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

type CublasHandle = *mut c_void;

const CUBLAS_STATUS_SUCCESS: i32 = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUDA_R_16F: i32 = 2;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut CublasHandle) -> i32;
    fn cublasDestroy_v2(handle: CublasHandle) -> i32;
    fn cublasSetStream_v2(handle: CublasHandle, stream: *mut c_void) -> i32;
    fn cublasGemmEx(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute_type: i32,
        algorithm: i32,
    ) -> i32;
}

pub(super) struct Cublas {
    handle: CublasHandle,
    stream: Arc<CudaStream>,
}

impl Cublas {
    pub(super) fn new(stream: Arc<CudaStream>) -> Result<Self, Box<dyn Error>> {
        stream.context().bind_to_thread()?;
        let mut handle = ptr::null_mut();
        check(unsafe { cublasCreate_v2(&mut handle) }, "cublasCreate_v2")?;
        if let Err(error) = check(
            unsafe { cublasSetStream_v2(handle, stream.cu_stream().cast()) },
            "cublasSetStream_v2",
        ) {
            unsafe {
                cublasDestroy_v2(handle);
            }
            return Err(error);
        }
        Ok(Self { handle, stream })
    }

    pub(super) fn linear(
        &self,
        input: &CudaView<'_, f16>,
        weight_oi: &CudaView<'_, f16>,
        output: &mut CudaViewMut<'_, f16>,
        m: usize,
        n: usize,
        k: usize,
        output_scale: f32,
        residual_scale: f32,
    ) -> Result<(), Box<dyn Error>> {
        if input.len() < m * k || weight_oi.len() < n * k || output.len() < m * n {
            return Err("cuBLAS linear view is smaller than its matrix dimensions".into());
        }
        let m = i32::try_from(m)?;
        let n = i32::try_from(n)?;
        let k = i32::try_from(k)?;
        let (input, _input_access) = input.device_ptr(&self.stream);
        let (weight, _weight_access) = weight_oi.device_ptr(&self.stream);
        let (output, _output_access) = output.device_ptr_mut(&self.stream);
        check(
            unsafe {
                cublasGemmEx(
                    self.handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    (&output_scale as *const f32).cast(),
                    weight as usize as *const c_void,
                    CUDA_R_16F,
                    k,
                    input as usize as *const c_void,
                    CUDA_R_16F,
                    k,
                    (&residual_scale as *const f32).cast(),
                    output as usize as *mut c_void,
                    CUDA_R_16F,
                    n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
        )
    }

    pub(super) fn linear_transposed_output(
        &self,
        input: &CudaView<'_, f16>,
        weight_oi: &CudaView<'_, f16>,
        output: &mut CudaViewMut<'_, f16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), Box<dyn Error>> {
        if input.len() < m * k || weight_oi.len() < n * k || output.len() < m * n {
            return Err(
                "cuBLAS transposed linear view is smaller than its matrix dimensions".into(),
            );
        }
        let m = i32::try_from(m)?;
        let n = i32::try_from(n)?;
        let k = i32::try_from(k)?;
        let one = 1.0_f32;
        let zero = 0.0_f32;
        let (input, _input_access) = input.device_ptr(&self.stream);
        let (weight, _weight_access) = weight_oi.device_ptr(&self.stream);
        let (output, _output_access) = output.device_ptr_mut(&self.stream);
        check(
            unsafe {
                cublasGemmEx(
                    self.handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    (&one as *const f32).cast(),
                    input as usize as *const c_void,
                    CUDA_R_16F,
                    k,
                    weight as usize as *const c_void,
                    CUDA_R_16F,
                    k,
                    (&zero as *const f32).cast(),
                    output as usize as *mut c_void,
                    CUDA_R_16F,
                    m,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx transposed output",
        )
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
        }
    }
}

fn check(status: i32, operation: &str) -> Result<(), Box<dyn Error>> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed with cuBLAS status {status}").into())
    }
}
