#include <x86intrin.h>

typedef struct {
    union {
    struct {float x,y,z,w;};
    struct {float r,g,b,a;};
    float e[4];
    __m128 sse;
    };
} vec4;

typedef struct {
    float x;
    float y;
    float z;
}vec3;

typedef struct {
    float x;
    float y;
}vec2;

typedef struct {
    int x;
    int y;
}vec2i;

typedef struct {
    vec2i min;
    vec2i max;
} Rect;

typedef struct {
    int has_normals;
    int has_uvs;
    vec4 verts[2];
    vec3 normals[2];
    vec2 uvs[2];
    vec4 c1[2];
    vec4 c2[2];
} Edge;

typedef struct {
    int has_normals;
    int has_uvs;
    vec4 verts[3];
    vec3 normals[3];
    vec2 uvs[3];
    vec4 c1[3];
    vec4 c2[3];
    uint8_t coarse_levels[3];
} Triangle;

typedef struct {
	int v;
	int uv;
	int n;
    int flags;
} MeshIndex;

typedef struct {
	float *vertices;
    float *objspace_vertices;
	float *uvs;
	float *normals;
	int *face_indices;
	int *uv_indices;
	int *normal_indices;
    uint8_t *coarse_levels;
    float *bbox;
}MeshData;

typedef struct {
  size_t width;
  size_t height;
  void *mem;
  size_t mem_size;
  float *r;
  float *g;
  float *b;
  float *a;
}Texture;

typedef struct {
    size_t width;
    size_t height;
    int perspective_correct;
    int uvspace;
    int projection;
    int checker_size;
    int wireframe;
    void *mem;
    size_t mem_size;
    Texture img;
	float *z;
}RenderContext;
