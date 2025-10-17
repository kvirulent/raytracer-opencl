// gradient.cl
// mango gradient

__kernel void gradient(__constant int* width, __constant int* height, __global float3* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (*width) + x;

    float fx = (float)x / (float)(*width);
    float fy = (float)y / (float)(*height);

    output[idx] = (float3)(fx, fy, 0.0f);
}
