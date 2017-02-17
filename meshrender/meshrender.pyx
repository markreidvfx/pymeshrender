from libc cimport math
from libc.string cimport memset
from cython cimport view
from Cython.Shadow import NULL
from numpy import clip
cimport cython

cdef extern from "core.h" nogil:
    ctypedef struct Point:
        float x
        float y
        float z
        float w

    ctypedef struct Triangle:
        Point v0
        Point v1
        Point v2

    ctypedef struct vec4:
        pass

    ctypedef struct vec2i:
        int x
        int y

    ctypedef struct Rect:
        vec2i min
        vec2i max

    ctypedef struct MeshIndex:
        int v;
        int uv;
        int n

    ctypedef struct Polygon:
        MeshIndex v0;
        MeshIndex v1;
        MeshIndex v2;

    ctypedef struct MeshData:
        float *vertices
        float *objspace_vertices
        float *uvs
        float *normals
        int *face_indices
        int *uv_indices
        int *normal_indices
        unsigned char *coarse_levels
        vec4 *bbox

    ctypedef struct Texture:
        int width
        int height
        void *mem
        float *r
        float *g
        float *b
        float *a

    ctypedef struct RenderContext:
        int width
        int height
        int perspective_correct
        int uvspace
        int projection
        int checker_size
        int wireframe
        void *mem
        int mem_size
        Texture img
        float *z
        float *nx
        float *ny
        float *nz

    cdef void setup_render_context(RenderContext *ctx, size_t width, size_t height)
    cdef void setup_texture_context(Texture *ctx, size_t width, size_t height)
    cdef void free_render_context(RenderContext *ctx)
    cdef void free_texture_context(Texture *ctx)
    cdef void under(RenderContext *ctx, Texture *tex)
    cdef void copy_texture(Texture *src, Texture *dst)
    cdef void grow_texture(Texture *ctx)
    cdef void grow_texture_new(Texture *ctx)
    cdef void load_packed_texture(Texture *ctx,
                     unsigned char *src,
                     size_t width, size_t height, int depth)

    cdef void load_texure(Texture *ctx,
                     unsigned char *src_r,
                     unsigned char *src_g,
                     unsigned char *src_b,
                     unsigned char *src_a,
                     size_t width, size_t height)

    cdef void resample_texture(Texture *src, Texture *dst)

    cdef void draw_edge(RenderContext *ctx, int* vert_indices, MeshData* mesh_data, Rect *clip, Texture *tex)
    cdef void draw_triangle_float(RenderContext *ctx, MeshIndex* mesh_indices, MeshData* mesh_data, Rect *clip, Texture *tex)
    cdef void draw_quad(RenderContext *ctx, MeshIndex* indices, const MeshData *mesh, Rect *clip, Texture *tex)
    cdef void draw_bbox(RenderContext *ctx, vec4 *bbox, const Rect *clip)

    cdef void texture_context_to_rgba(Texture *ctx, unsigned char *dst)
    cdef void texture_context_to_rgba16(Texture *ctx, unsigned char *dst)

@cython.boundscheck(False)
def perpsective_divide(float [:,:] values):
    assert values.shape[1] == 4
    cdef int i
    cdef float w
    cdef float *v = &values[0][0]
    with nogil:
        for i in range(values.shape[0]):
            w = 1.0/v[3]

            v[0] *= w
            v[1] *= w
            v[2] *= w
            v += 4

def viewport_matrix(float width, float height):
    m = view.array(shape=(4, 4), itemsize=sizeof(float), format="f")
    for y in range(4):
        for x in range(4):
            m[y][x] = 0.0

    m[0][0] = +width/2.0
    m[0][3] = +width/2.0

    m[1][1] = -height/2.0
    m[1][3] = +height/2.0

    m[2][2] = 1.0
    m[3][3] = 1.0

    return m

cdef class Mesh(object):
    cdef public int[:] face_counts
    cdef public int[:] face_indices
    cdef public int[:] uv_indices
    cdef public int[:] normal_indices

    cdef public float [:,:] vertices
    cdef public float [:,:] objspace_vertices
    cdef public float [:,:] uvs
    cdef public float [:,:] normals

    cdef public float[:,:] bbox
    cdef public unsigned char[:] coarse_levels

    def __cinit__(self):
        self.face_counts = None
        self.face_indices = None
        self.uv_indices = None
        self.normal_indices = None
        self.vertices = None
        self.objspace_vertices = None
        self.normals = None
        self.coarse_levels = None
        self.bbox = None

    def from_meshsmooth(self, mesh, uvchannel, normalchannel):

        self.face_counts = mesh.face_counts
        self.face_indices = mesh.vertices.indices
        self.vertices = mesh.vertices.values
        self.coarse_levels = mesh.coarse_levels

        self.uv_indices = uvchannel.indices
        self.uvs = uvchannel.values
        if normalchannel:
            self.normal_indices = normalchannel.indices
            self.normals = normalchannel.values

    cdef MeshData get_data(self):
        cdef MeshData data
        data.normal_indices = NULL
        data.normals = NULL

        data.uvs = NULL
        data.uv_indices = NULL

        data.vertices = &self.vertices[0][0]
        data.objspace_vertices = &self.objspace_vertices[0][0]
        data.face_indices = &self.face_indices[0]

        data.coarse_levels = NULL
        data.bbox = NULL

        if not self.uv_indices is None and not self.uvs is None:
            data.uvs = &self.uvs[0][0]
            data.uv_indices = &self.uv_indices[0]

        if not self.normal_indices is None and not self.normals is None:
            data.normal_indices = &self.normal_indices[0]
            data.normals = &self.normals[0][0]

        if not self.coarse_levels is None:
            data.coarse_levels = &self.coarse_levels[0]

        if not self.bbox is None:
            data.bbox = <vec4*> &self.bbox[0][0]

        return data

cdef MeshIndex get_mesh_indices(MeshData mesh, int index) nogil:
    cdef MeshIndex i
    i.v = mesh.face_indices[index]
    if  mesh.uv_indices:
        i.uv = mesh.uv_indices[index]
    if mesh.normal_indices:
        i.n = mesh.normal_indices[index]
    return i

cdef float clampf(float value, float mn, float mx) nogil:
    return math.fmin(math.fmax(value, mn), mx);

cdef unsigned int roundup(unsigned int a, unsigned int b):
    return  (a + b-1)/b * b

cdef class MeshTexture(object):

    cdef Texture ctx

    def __cinit__(self):
        self.ctx.mem = NULL;

    def __dealloc__(self):
        free_texture_context(&self.ctx)

    def __init__(self, unsigned char[::1] src,
                       unsigned int width, unsigned int height,
                       unsigned int depth = 8):

        with nogil:
            setup_texture_context(&self.ctx, width, height)
            load_packed_texture(&self.ctx, &src[0], width, height, depth)

    def grow(self, unsigned int amount=1):
        with nogil:
            for i in range(amount):
                grow_texture(&self.ctx)

    def resample(self, unsigned int width, unsigned int height):
        cdef MeshTexture dst = MeshTexture.__new__(MeshTexture)
        with nogil:
            setup_texture_context(&dst.ctx, width, height)
            resample_texture(&self.ctx, &dst.ctx)
        return dst

    property rgba:
        @cython.boundscheck(False)
        def __get__(self):
            cdef unsigned char[:,:,:] data = view.array(shape=(self.height, self.width, 4),
                                                        itemsize=sizeof(unsigned char), format="B")

            cdef unsigned char *d = <unsigned char *>&data[0][0][0]

            with nogil:
                texture_context_to_rgba(&self.ctx, d)

            return data
    property r:
        @cython.boundscheck(False)
        def __get__(self):
            size = self.ctx.width * self.ctx.height
            cdef view.array data  = <float [:size]>self.ctx.r
            return data

    property g:
        @cython.boundscheck(False)
        def __get__(self):
            size = self.ctx.width * self.ctx.height
            cdef view.array data  = <float [:size]>self.ctx.g
            return data

    property b:
        @cython.boundscheck(False)
        def __get__(self):
            size = self.ctx.width * self.ctx.height
            cdef view.array data  = <float [:size]>self.ctx.b
            return data

    property a:
        @cython.boundscheck(False)
        def __get__(self):
            size = self.ctx.width * self.ctx.height
            cdef view.array data  = <float [:size]>self.ctx.a
            return data

    property rgba16:
        @cython.boundscheck(False)
        def __get__(self):
            cdef unsigned short[:,:,:] data = view.array(shape=(self.height, self.width, 4),
                                                        itemsize=sizeof(unsigned short), format="H")

            cdef unsigned char *d = <unsigned char *>&data[0][0][0]

            with nogil:
                texture_context_to_rgba16(&self.ctx, d)

            return data

    property width:
        def __get__(self):
            return self.ctx.width
    property height:
        def __get__(self):
            return self.ctx.height

cdef class MeshRenderer(object):
    cdef public unsigned int width
    cdef public unsigned int height
    cdef RenderContext ctx
    cdef public int bbox

    def __cinit__(self):
        self.ctx.mem = NULL;
        self.ctx.img.mem = NULL;
        self.bbox = 0;

    def __dealloc__(self):
        free_render_context(&self.ctx)

    def __init__(self, unsigned int width, unsigned int height):
        self.resize(width, height)
        self.ctx.checker_size = 50
        self.ctx.uvspace = 1
        self.ctx.perspective_correct = 1

    def resize(self, unsigned int width, unsigned int height):
        # width and height always need to be multiples of 4
        width = roundup(width, 4);
        #height =  roundup(height, 4);

        self.width = width
        self.height = height
        with nogil:
            setup_render_context(&self.ctx, width, height);

        self.clear()

    @cython.boundscheck(False)
    def clear(self):
        cdef size_t size = self.width * self.height * sizeof(float)
        cdef int i
        with nogil:
            # channels are one big malloc
            memset(self.ctx.mem, 0, self.ctx.mem_size)
            for i in range(self.width * self.height):
                self.ctx.z[i] = 1.0

    @cython.boundscheck(False)
    def render_mesh(self, Mesh mesh not None, rect = None, MeshTexture texture = None):

        cdef MeshIndex mesh_index[4]
        cdef int current_index = 0
        cdef int face = 0
        cdef int i
        cdef int j

        cdef Texture *tex = NULL

        if texture:
            tex = &texture.ctx

        cdef MeshData mesh_data = mesh.get_data()
        cdef Rect clip

        clip.min.x = 0
        clip.min.y = 0

        clip.max.x = self.width - 1
        clip.max.y = self.height -1

        if rect:
            clip.min.x = rect[0][0]
            clip.min.y = rect[0][1]
            clip.max.x = rect[1][0]
            clip.max.y = rect[1][1]

        with nogil:

            if self.bbox and mesh_data.bbox and not self.ctx.uvspace:
                draw_bbox(&self.ctx, mesh_data.bbox, &clip)

            for i in range(mesh.face_counts.shape[0]):
                face = mesh.face_counts[i]

                if face < 2:
                    current_index += face
                    continue

                if face == 4:
                    for j in range(4):
                        mesh_index[j] = get_mesh_indices(mesh_data, current_index + j)
                    draw_quad(&self.ctx, mesh_index, &mesh_data, &clip, tex)
                else:
                    # triangulate
                    mesh_index[0] = get_mesh_indices(mesh_data, current_index)
                    for j in range(1, face-1):
                        mesh_index[1] = get_mesh_indices(mesh_data, current_index + j + 0)
                        mesh_index[2] = get_mesh_indices(mesh_data, current_index + j + 1)
                        draw_triangle_float(&self.ctx, mesh_index, &mesh_data, &clip, tex)

                current_index += face

    def under(self, MeshTexture tex not None):
        with nogil:
            under(&self.ctx, &tex.ctx)

    def grow(self, unsigned int amount=16):
        with nogil:
            # for i in range(amount):
            #     grow_texture(&self.ctx.img)
            grow_texture_new(&self.ctx.img)

    def to_texture(self, width=None, height=None):

        cdef MeshTexture tex = MeshTexture.__new__(MeshTexture)
        cdef unsigned int tex_width
        cdef unsigned int tex_height

        if width or height:
            if not width:
                tex_width = self.width
            else:
                tex_width = width

            if not height:
                tex_height = self.height
            else:
                tex_height = height

            with nogil:
                setup_texture_context(&tex.ctx, tex_width, tex_height)
                resample_texture(&self.ctx.img, &tex.ctx)
        else:

            with nogil:
                copy_texture(&self.ctx.img, &tex.ctx)

        return tex

    property viewport_matrix:
        def __get__(self):
            return viewport_matrix(self.width - 1, self.height - 1)

    property rgba:
        @cython.boundscheck(False)
        def __get__(self):
            cdef unsigned char[:,:,:] data = view.array(shape=(self.height,self.width, 4), itemsize=sizeof(unsigned char), format="B")
            cdef int size = self.width * self.height
            cdef int i

            cdef unsigned char *d = <unsigned char *>&data[0][0][0]

            with nogil:
                texture_context_to_rgba(&self.ctx.img, d)

            return bytearray(data)

    property rgb:
        @cython.boundscheck(False)
        def __get__(self):
            cdef unsigned char[:,:,:] data = view.array(shape=(self.height,self.width, 3), itemsize=sizeof(unsigned char), format="B")
            cdef int size = self.width * self.height
            cdef int i

            cdef unsigned char *d = <unsigned char *>&data[0][0][0]

            with nogil:
                for i in range(size):
                    d[0] =  <unsigned char> clampf(self.ctx.img.r[i] * 255.0, 0, 255.0)
                    d += 1
                    d[0] =  <unsigned char> clampf(self.ctx.img.g[i] * 255.0, 0, 255.0)
                    d += 1
                    d[0] =  <unsigned char> clampf(self.ctx.img.b[i] * 255.0, 0, 255.0)
                    d += 1

            return bytearray(data)

    property rgba16:
        @cython.boundscheck(False)
        def __get__(self):
            cdef unsigned short[:,:,:] data = view.array(shape=(self.height, self.width, 4),
                                                        itemsize=sizeof(unsigned short), format="H")

            cdef unsigned char *d = <unsigned char *>&data[0][0][0]

            with nogil:
                texture_context_to_rgba16(&self.ctx.img, d)

            return data


    property checker_size:
        def __set__(self, int value):
            self.ctx.checker_size = value

    property bake_projection:
        def __set__(self, int value):
            self.ctx.projection = value

    property uvspace:
        def __set__(self, int value):
          self.ctx.uvspace = value
        def __get__(self):
            return self.ctx.uvspace == 1

    property wireframe:
        def __set__(self, int value):
            self.ctx.wireframe = value
        def __get__(self):
            return self.ctx.wireframe == 1

    property perspective_correct:
        def __set__(self, int value):
            self.ctx.perspective_correct = value
        def __get__(self):
            return self.ctx.perspective_correct == 1
