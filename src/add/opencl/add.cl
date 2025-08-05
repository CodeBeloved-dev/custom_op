__kernel void add(__global const float* a, __global const float* b, __global float* out, long n) {
    int idx = get_global_id(0);
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
