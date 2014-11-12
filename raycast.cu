#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define DIM_SHARED 10

// Color names will be a bit better than numbers
#define BLUE 2
#define RED 1
#define NO_COLOR 0

// Just some comfort
#define bool int
#define true 1
#define false 0
#define boolval(x) ((x == true) ? "true" : "false")

// Textures declaration for raycast_gpu_texture
texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> tex_data;
texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> tex_region;

// Stack for the serial region growing
typedef struct{
    int size;
    int buffer_size;
    int3* pixels;
} my_stack_t;

my_stack_t* new_stack(){
    my_stack_t* stack = (my_stack_t*)malloc(sizeof(my_stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (int3*)malloc(sizeof(int3)*1024);

    return stack;
}

void push(my_stack_t* stack, int3 p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        int3* temp = stack->pixels;
        stack->pixels = (int3*)malloc(sizeof(int3)*stack->buffer_size);
        memcpy(stack->pixels, temp, sizeof(int3)*stack->buffer_size/2);
        free(temp);

    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}

int3 pop(my_stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}

// float3 utilities
__device__ __host__ float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__device__ __host__ float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

__device__ __host__ float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

__device__ __host__ float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
}


// Prints CUDA device properties
void print_properties(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

    cudaDeviceProp p;
    cudaSetDevice(0);
    cudaGetDeviceProperties (&p, 0);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Name: %s\n" , p.name);
    printf("\n\n");
}


// Fills data with values
unsigned char func(int x, int y, int z){
    unsigned char value = rand() % 20;

    int x1 = 300;
    int y1 = 400;
    int z1 = 100;
    float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 100){
        value  = 30;
    }

    x1 = 100;
    y1 = 200;
    z1 = 400;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 50){
        value = 50;
    }

    if(x > 200 && x < 300 && y > 300 && y < 500 && z > 200 && z < 300){
        value = 45;
    }
    if(x > 0 && x < 100 && y > 250 && y < 400 && z > 250 && z < 400){
        value =35;
    }
    return value;
}

unsigned char* create_data(){
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM);

    for(int i = 0; i < DATA_DIM; i++){
        for(int j = 0; j < DATA_DIM; j++){
            for(int k = 0; k < DATA_DIM; k++){
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM+ k]= func(k,j,i);
            }
        }
    }

    return data;
}

// Checks if position is inside the volume (float3 and int3 versions)
__device__ __host__ int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

__device__ __host__ int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

// Indexing function (note the argument order)
__device__ __host__ int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Indexing function (note the argument order)
__device__ int index_shared(int z, int y, int x) {
    return (z % 10)*DIM_SHARED*DIM_SHARED + (y % 10)*DIM_SHARED + (x % 10);
}

// Trilinear interpolation
__device__ __host__ float value_at(float3 pos, unsigned char* data)
{
    if(!inside(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;


    return c0;
}

// Trilinear interpolation using a texture reference
__device__ float value_at_tex(float3 pos, texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> tex)
{
    if(!inside(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx * tex1Dfetch(tex, index(z,y,x)) + (1-rx) * tex1Dfetch(tex, index(z,y,x_u));
	float a1 = rx * tex1Dfetch(tex, index(z,y_u,x)) + (1-rx) * tex1Dfetch(tex, index(z,y_u,x_u));
	float a2 = rx * tex1Dfetch(tex, index(z_u,y,x)) + (1-rx) * tex1Dfetch(tex, index(z_u,y,x_u));
	float a3 = rx * tex1Dfetch(tex, index(z_u,y_u,x)) + (1-rx) * tex1Dfetch(tex, index(z_u,y_u,x_u));

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;


    return c0;
}

// Check if two values are similar, threshold can be changed.
__device__ __host__ int similar(unsigned char* data, int3 a, int3 b) {
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

    int i = abs(va-vb) < 1;
    return i;
}

// Serial ray casting
unsigned char* raycast_serial(unsigned char* data, unsigned char* region){
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    // For each pixel
    for(int y = -(IMAGE_DIM/2); y < (IMAGE_DIM/2); y++){
        for(int x = -(IMAGE_DIM/2); x < (IMAGE_DIM/2); x++){

            // Find the ray for this pixel
            float3 screen_center = add(camera, forward);
            float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
            ray = add(ray, scale(camera, -1));
            ray = normalize(ray);
            float3 pos = camera;

            // Move along the ray, we stop if the color becomes completely white,
            // or we've done 5000 iterations (5000 is a bit arbitrary, it needs
            // to be big enough to let rays pass through the entire volume)
            int i = 0;
            float color = 0;
            while(color < 255 && i < 5000){
                i++;
                pos = add(pos, scale(ray, step_size));          // Update position
                int r = value_at(pos, region);                  // Check if we're in the region
                color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
            }

            // Write final color to image
            image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
        }
    }

    return image;
}

__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	x = (x < IMAGE_DIM / 2) ? -x : x - IMAGE_DIM / 2;
	y = (y < IMAGE_DIM / 2) ? -y : y - IMAGE_DIM / 2;

	// Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

	// Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region)
{
	printf("\n\n=== STARTING RAYCASTING ===\n");
	unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);
    
    // Allocating and copying to GPU mem
    printf("\n+ Allocating and copying data to GPU mem\n");
    int dataSize = DATA_DIM * DATA_DIM * DATA_DIM * sizeof(unsigned char);
    int imageSize = IMAGE_DIM * IMAGE_DIM * sizeof(unsigned char);
    unsigned char *dev_data, *dev_image, *dev_region;
	cudaMalloc((void**)&dev_data, dataSize);
	printf("\tError at cudalloc dev_data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_image, imageSize);
	printf("\tError at cudalloc dev_image: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_region, dataSize);
	printf("\tError at cudalloc dev_region: %s\n", cudaGetErrorString(cudaGetLastError()));
	
	cudaMemcpy(dev_data, data, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_image, image, imageSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for image: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_region, region, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for region: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    // Launching kernel
    printf("\n+ Launching kernel...\n");
    dim3 dimBlock(64, 64, 1);
	dim3 dimThread(8, 8, 1);
	raycast_kernel<<<dimBlock, dimThread>>>(dev_data, dev_image, dev_region);
	printf("\tKernel returned: %s\n", cudaGetErrorString(cudaGetLastError()));
	// Getting the image back from GPU mem
	printf("\tFetching image from GPU mem.\n");
	cudaMemcpy(image, dev_image, imageSize, cudaMemcpyDeviceToHost);
	printf("\tImage has been retrieved from CUDA mem: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("\n\n=== ------------------- ===\n");
    return image;
}

__global__ void raycast_kernel_texture(unsigned char* image)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	x = (x < IMAGE_DIM / 2) ? -x : x - IMAGE_DIM / 2;
	y = (y < IMAGE_DIM / 2) ? -y : y - IMAGE_DIM / 2;

	// Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

	// Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at_tex(pos, tex_region);                  // Check if we're in the region
        color += value_at_tex(pos, tex_data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}

unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region)
{
	printf("\n\n=== STARTING TEXTURED RAYCASTING ===\n");
	unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

	// Allocating and copying memory to GPU memory
	printf("\n+ Allocating and copying data to GPU mem\n");
    size_t dataSize = DATA_DIM * DATA_DIM * DATA_DIM * sizeof(unsigned char);
    size_t imageSize = IMAGE_DIM * IMAGE_DIM * sizeof(unsigned char);
    unsigned char *dev_data, *dev_image, *dev_region;
	cudaMalloc((void**)&dev_data, dataSize);
	printf("\tAllocating dev_data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_image, imageSize);
	printf("\tAllocating dev_image: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_region, dataSize);
	printf("\tAllocating dev_region: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(dev_data, data, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_image, image, imageSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for image: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_region, region, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for region: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Binding textures
	printf("\n+ Binding textures\n");
	cudaBindTexture(0, tex_data, dev_data, dataSize);
	printf("\tBound data texture: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaBindTexture(0, tex_region, dev_region, dataSize);
	printf("\tBound region texture: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Launching kernel
    printf("\n+ Launching kernel...\n");
    dim3 dimBlock(64, 64, 1);
	dim3 dimThread(8, 8, 1);
	raycast_kernel_texture<<<dimBlock, dimThread>>>(dev_image);
	printf("\tKernel returned: %s\n", cudaGetErrorString(cudaGetLastError()));
	// Getting the image back from GPU mem
	printf("\tFetching image from GPU mem.\n");
	cudaMemcpy(image, dev_image, imageSize, cudaMemcpyDeviceToHost);
	printf("\tImage has been retrieved from CUDA mem: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("\n\n=== ---------------------------- ===\n");
    return image;
}

// Serial region growing, same algorithm as in assignment 2
unsigned char* grow_region_serial(unsigned char* data){
    unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);

    my_stack_t* stack = new_stack();

    int3 seed = {.x=50, .y=300, .z=300};
    push(stack, seed);
    region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = RED;

    int dx[6] = {-1,1,0,0,0,0};
    int dy[6] = {0,0,-1,1,0,0};
    int dz[6] = {0,0,0,0,-1,1};

    while(stack->size > 0){
        int3 pixel = pop(stack);
        for(int n = 0; n < 6; n++){
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if(!inside(candidate)){
                continue;
            }

            if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
                continue;
            }

            if(similar(data, pixel, candidate)){
                push(stack, candidate);
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 1;
            }
        }
    }

    return region;
}

__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int z = (blockIdx.z * blockDim.z + threadIdx.z);
	int idx = index(z, y, x);

	if (region[idx] == RED)
	{
		region[idx] = BLUE;
		int3 voxel = {.x = x, .y = y, .z = z};
		int3 neighbors[] = {
			{.x = x + 1, .y = y, .z = z},
			{.x = x - 1, .y = y, .z = z},
			{.x = x, .y = y + 1, .z = z},
			{.x = x, .y = y - 1, .z = z},
			{.x = x, .y = y, .z = z + 1},
			{.x = x, .y = y, .z = z - 1}
		};

		for (int i = 0; i < 6; i++)
		{
			if (inside(neighbors[i]))
			{
				if (region[index(neighbors[i].z, neighbors[i].y, neighbors[i].x)] == NO_COLOR && similar(data, voxel, neighbors[i]))
				{
					region[index(neighbors[i].z, neighbors[i].y, neighbors[i].x)] = RED;
					*finished = false;
				}
			}
		}
	}
}

unsigned char* grow_region_gpu(unsigned char* data)
{
	printf("\n\n=== STARTING GROW REGION ===\n");
	int dataSize = DATA_DIM * DATA_DIM * DATA_DIM * sizeof(unsigned char);
	bool* finished = (bool*)malloc(sizeof(bool));

	// Allocating region and planting seed
	unsigned char* region = (unsigned char*)calloc(DATA_DIM * DATA_DIM * DATA_DIM, sizeof(unsigned char));
	int3 seed = {.x=50, .y=300, .z=300};
    region[index(seed.z, seed.y, seed.x)] = RED;
    printf("Seed has been set.\n");

	// Allocating then copying data and region to GPU memory
	printf("\n+ Allocating and copying data to GPU mem\n");
	unsigned char *dev_data, *dev_region;
	int* dev_finished;
	cudaMalloc((void**)&dev_data, dataSize);
	printf("\tError at cudalloc dev_data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_region, dataSize);
	printf("\tError at cudalloc dev_region: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_finished, sizeof(int*));
	printf("\tError at cudalloc dev_finished: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(dev_data, data, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_region, region, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for region: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_finished, finished, sizeof(bool*), cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for finished: %s\n", cudaGetErrorString(cudaGetLastError()));

	// 8 blocks of 64 threads
	dim3 dimBlock(64, 64, 64);
	dim3 dimThread(8, 8, 8);

	printf("\n+ Starting kernel calls\n");
	do {
		*finished = true;
		cudaMemcpy(dev_finished, finished, sizeof(bool*), cudaMemcpyHostToDevice);
		region_grow_kernel<<<dimBlock, dimThread>>>(dev_data, dev_region, dev_finished);
		cudaMemcpy(finished, dev_finished, sizeof(bool*), cudaMemcpyDeviceToHost);
	} while (*finished == false);

	// Getting region back from GPU
	printf("\n+ Fetching region from GPU mem.\n");
	cudaMemcpy(region, dev_region, dataSize, cudaMemcpyDeviceToHost);
	printf("\tRegion has been retrieved from CUDA mem: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("\n\n=== -------------------- ===\n");
    return region;
}



__global__ void region_grow_kernel_shared(unsigned char* data, unsigned char* region, bool* finished)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int z = (blockIdx.z * blockDim.z + threadIdx.z);
    int idx = index_shared(z, y, x);

    // Each thread loads its neighbours in the shared memory
    extern __shared__ unsigned char shared_region[];
    int dx[] = {-1, +1, 0, 0, 0, 0};
    int dy[] = {0, 0, -1, +1, 0, 0};
    int dz[] = {0, 0, 0, 0, -1, +1};

    for (int i = 0; i < 6; i++) {
        int3 neighbour_abs = {.x = x + dx[i], .y = y + dy[i], .z = z + dz[i]};
        if (inside(neighbour_abs))
            shared_region[index_shared(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)] = region[index(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)];
    }

	__syncthreads();

    // Region growing per se
    int3 voxel = {.x = x, .y = y, .z = z};
	if (shared_region[idx] == RED)
    {
        shared_region[idx] = BLUE;
        for (int i = 0; i < 6; i++)
        {
            int3 neighbour_abs = {.x = x + dx[i], .y = y + dy[i], .z = z + dz[i]};
            if (inside(neighbour_abs)) {
                if (shared_region[index_shared(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)] == NO_COLOR && similar(data, voxel, neighbour_abs))
                {
                    shared_region[index_shared(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)] = RED;
                    *finished = false;
                }
            }
        }
    }

    // Each thread copies its neighbours back to the global memory
    for (int i = 0; i < 6; i++) {
        int3 neighbour_abs = {.x = x + dx[i], .y = y + dy[i], .z = z + dz[i]};
        if (inside(neighbour_abs))
            region[index(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)] = shared_region[index_shared(neighbour_abs.z, neighbour_abs.y, neighbour_abs.x)];
    }
}

unsigned char* grow_region_gpu_shared(unsigned char* data)
{
	printf("\n\n=== STARTING SHARED GROW REGION ===\n");
	int dataSize = DATA_DIM * DATA_DIM * DATA_DIM * sizeof(unsigned char);
	bool* finished = (bool*)malloc(sizeof(bool));

	// Allocating region and planting seed
	unsigned char* region = (unsigned char*)calloc(DATA_DIM * DATA_DIM * DATA_DIM, sizeof(unsigned char));
	int3 seed = {.x=50, .y=300, .z=300};
    region[index(seed.z, seed.y, seed.x)] = RED;
    printf("Seed has been set.\n");

	// Allocating then copying data and region to GPU memory
	printf("\n+ Allocating and copying data to GPU mem\n");
	unsigned char *dev_data, *dev_region;
	int* dev_finished;
	cudaMalloc((void**)&dev_data, dataSize);
	printf("\tError at cudalloc dev_data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_region, dataSize);
	printf("\tError at cudalloc dev_region: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&dev_finished, sizeof(int));
	printf("\tError at cudalloc dev_finished: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(dev_data, data, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for data: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_region, region, dataSize, cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for region: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dev_finished, finished, sizeof(bool), cudaMemcpyHostToDevice);
	printf("\tCopied host mem to GPU memory for finished: %s\n", cudaGetErrorString(cudaGetLastError()));

	// 8 blocks of 64 threads
	int n_threadsX = 8, n_threadsY = 8, n_threadsZ = 8;
	dim3 dimBlock(64, 64, 64);
	dim3 dimThread(n_threadsX, n_threadsY, n_threadsZ);
	int shareMemSize = (n_threadsX + 2) * (n_threadsY + 2) * (n_threadsZ + 2) * sizeof(unsigned char);

	printf("\n+ Starting kernel calls\n");
	do {
		*finished = true;
		cudaMemcpy(dev_finished, finished, sizeof(bool), cudaMemcpyHostToDevice);
		//printf("\tCopying Finished to GPU mem: %s\n", cudaGetErrorString(cudaGetLastError()));
		region_grow_kernel_shared<<<dimBlock, dimThread, shareMemSize>>>(dev_data, dev_region, dev_finished); // Third parameter declares the shared mem size
		//printf("\tLast kernel call: %s\n", cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(finished, dev_finished, sizeof(bool), cudaMemcpyDeviceToHost);
		//printf("\tGetting Finished back to CPU: %s\n", cudaGetErrorString(cudaGetLastError()));
	} while (*finished == false);
    printf("\tKernel calls finished: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Getting region back from GPU
	printf("\n+ Fetching region from GPU mem.\n");
	cudaMemcpy(region, dev_region, dataSize, cudaMemcpyDeviceToHost);
	printf("\tRegion has been retrieved from CUDA mem: %s\n", cudaGetErrorString(cudaGetLastError()));

	printf("\n\n=== --------------------------- ===\n");

	return region;
}


int main(int argc, char** argv){

    print_properties();

    printf("=============\n");
    unsigned char* data = create_data();
    printf("Data created.\n");
    printf("=============\n");

    unsigned char* region = grow_region_gpu_shared(data);
    printf("Region growing finished.\n");

    //unsigned char* image = raycast_serial(data, region);
    unsigned char* image = raycast_gpu_texture(data, region);
    printf("Raycast finished.\n");

    write_bmp(image, IMAGE_DIM, IMAGE_DIM);
}
