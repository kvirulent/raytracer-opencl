// intersect_rs.cl
// calculate the intersect of a ray and sphere

__kernel void intersect_rs(
    __constant int* width, 
    __constant int* height, 
    __constant float* fov,
    __constant float3* sphere_origin, 
    __constant float* sphere_radius,
    __global float3* output) 
{
    // get the pixel we are calculating for
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (*width) + x;

    // calculate the ray direction for this pixel
    float ray_vector_x = x + 0.5 - *width / 2;
    float ray_vector_y = y + 0.5 - *height / 2;
    float ray_vector_z = -*height / (2.f * tan(*fov/2.f));
    float3 ray_direction = normalize((float3)(ray_vector_x, ray_vector_y, ray_vector_z));
    float3 ray_origin = (float3)(0.f, 0.f, 0.f);

    // a vector extending from the ray's origin toward the sphere's origin
    float3 L = *sphere_origin - ray_origin;

    // project the vector L onto the ray
    float tca = dot(L, ray_direction);

    // the direction of the ray and the direction of L point in oppsite directions,
    // thus the sphere is behind the ray and is not visible
    if (tca < 0) {
        output[idx] = (float3)(0.2, 0.7, 0.8); // sphere is behind the ray, color as bg
        return;
    }

    // calculate the distance between the center of the sphere
    // and the end of the projected vector L
    float d2 = dot(L,L) - tca*tca;
    float r2 = (*sphere_radius) * (*sphere_radius);

    // if the distance between the sphere and the vector L is greater
    // than the sphere's radius, the ray did not hit the sphere
    if (d2 > r2) {
        output[idx] = (float3)(0.2, 0.7, 0.8); // tca is not within the radius of the sphere (miss), color as bg
        return;
    }

    // distance to the intersections from d2
    float thc = sqrt(r2 - d2);

    // positions of the intersections
    float t0 = tca - thc;
    float t1 = tca + thc;

    if (t0 < 0.0f) t0 = t1; // the first intersection is behind, use the second
    if (t0 < 0.0f) {
        output[idx] = (float3)(0.2f, 0.7, 0.8f); // both intersections are behind, color as bg
        return; 
    } else {
        output[idx] = (float3)(0.4, 0.4, 0.3); // at least 1 intersection is not behind, color as sphere
    }
    
}