#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdint.h>
#include "core.h"
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MIN3(a,b,c) MIN(a, MIN(b, c))
#define MAX3(a,b,c) MAX(a, MAX(b, c))

#define ARRAY_SIZE( arr) (sizeof(arr)/sizeof(0[arr]))


void free_render_context(RenderContext *ctx)
{
    if(ctx->mem)
        free(ctx->mem);
}

void free_texture_context(Texture *ctx)
{
    if(ctx->mem)
        free(ctx->mem);
}

void setup_texture_context(Texture *ctx, size_t width, size_t height)
{
    int nb_channels = 4;
    int channel_size = width * height * sizeof(float);

    int mem_size = (channel_size *nb_channels) + 15;

    ctx->width = width;
    ctx->height = height;

    if (ctx->mem)
        free_texture_context(ctx);

    void *mem = malloc(mem_size);
    if (!mem)
        printf("unable to malloc texture\n");
    assert(mem);
    ctx->mem = mem;
    ctx->mem_size = mem_size;
    uint8_t *ptr = (uint8_t*) (((uintptr_t)mem+15) & ~ (uintptr_t)0x0F);
    ctx->r = (float*)ptr;
    ptr += channel_size;
    ctx->g = (float*)ptr;
    ptr += channel_size;
    ctx->b = (float*)ptr;
    ptr += channel_size;
    ctx->a = (float*)ptr;
}

void setup_render_context(RenderContext *ctx, size_t width, size_t height)
{
    int nb_channels = 4+1;
    int channel_size = width * height * sizeof(float);
    int mem_size = (channel_size *nb_channels) + 15;

    ctx->width = width;
    ctx->height = height;
    ctx->img.width = width;
    ctx->img.height = height;

    if (ctx->mem)
        free_render_context(ctx);

    ctx->mem = NULL;
    //malloc 16 aligned
    void *mem = malloc(mem_size);
    assert(mem);
    ctx->mem = mem;
    ctx->mem_size = mem_size;
    uint8_t *ptr = (uint8_t*) (((uintptr_t)mem+15) & ~ (uintptr_t)0x0F);

    ctx->img.r = (float*)ptr;
    ptr += channel_size;
    ctx->img.g = (float*)ptr;
    ptr += channel_size;
    ctx->img.b = (float*)ptr;
    ptr += channel_size;
    ctx->img.a = (float*)ptr;
    ptr += channel_size;
    ctx->z = (float*)ptr;
}


void copy_texture(Texture *src, Texture *dst)
{
    int nb_channels = 4;
    int channel_size = src->width * src->height * sizeof(float);
    int mem_size = channel_size * nb_channels;
    setup_texture_context(dst, src->width, src->height);
    memcpy(dst->r, src->r, mem_size);
}

void load_packed_texture(Texture *ctx,
                         uint8_t *src, size_t width, size_t height, int depth)
{
    int x = 0;
    int y = 0;

    int y_max = MIN(height, ctx->height);
    int x_max = MIN(width,  ctx->width);

    float max_pixel = 1.0f / ((1 << depth) - 1);
    int bytes_size = depth/8;
    int channels =4;

    for (y = 0; y < y_max; y++) {

        int offset = y * (ctx->width);
        float *r = ctx->r + offset;
        float *g = ctx->g + offset;
        float *b = ctx->b + offset;
        float *a = ctx->a + offset;


        int src_offset = (y * (width)) * bytes_size * channels;
        uint8_t *pixel = src + src_offset;
        uint16_t *pix16 = (uint16_t*)pixel;

        for (x =0; x < x_max; x++) {

            if (depth == 8) {
                *b = *pixel * max_pixel;
                pixel++;
                *g = *pixel * max_pixel;
                pixel++;
                *r = *pixel * max_pixel;
                pixel++;
                *a = *pixel * max_pixel;
                pixel++;
            } else if (depth == 16) {
                *b = *pix16 * max_pixel;
                pix16++;
                *g = *pix16 * max_pixel;
                pix16++;
                *r = *pix16 * max_pixel;
                pix16++;
                *a = *pix16 * max_pixel;
                pix16++;

            }

            r++;
            g++;
            b++;
            a++;
        }
    }
}

void load_texure(Texture *ctx,
                 uint8_t *src_r,
                 uint8_t *src_g,
                 uint8_t *src_b,
                 uint8_t *src_a,
                 size_t width, size_t height)
{

    int x = 0;
    int y = 0;

    int y_max = MIN(height, ctx->height);
    int x_max = MIN(width,  ctx->width);
    for (y = 0; y < y_max; y++) {

        int offset = y * (ctx->width);
        float *r = ctx->r + offset;
        float *g = ctx->g + offset;
        float *b = ctx->b + offset;
        float *a = ctx->a + offset;

        int bytes_size = 1;
        int src_offset = (y * (width)) * bytes_size;

        uint8_t *s_r = src_r + src_offset;
        uint8_t *s_g = src_g + src_offset;
        uint8_t *s_b = src_b + src_offset;
        uint8_t *s_a = src_a + src_offset;

        for (x =0; x < x_max; x++) {

            *r = *s_r / 255.0f;
            *g = *s_g / 255.0f;
            *b = *s_b / 255.0f;
            *a = *s_a / 255.0f;

            s_r += bytes_size;
            s_g += bytes_size;
            s_b += bytes_size;
            s_a += bytes_size;
            r++;
            g++;
            b++;
            a++;
        }
    }
}

static inline vec4 make_vec4(float x, float y, float z, float w)
{
    vec4 r;
    r.x = x;
    r.y = y;
    r.z = z;
    r.w = w;
    return r;
}

static inline vec4 add_vec4(vec4 a, vec4 b)
{

    vec4 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    r.w = a.w + b.w;
    return r;
}

static inline vec4 div_vec4(vec4 a, vec4 b)
{
    vec4 r;
    r.x = a.x / b.x;
    r.y = a.y / b.y;
    r.z = a.z / b.z;
    r.w = a.w / b.w;
    return r;
}

static inline vec4 div_vec4_f(vec4 a, float b)
{
    return div_vec4(a, make_vec4(b,b,b,b));
}



void under(RenderContext *rctx, Texture *tex)
{
    Texture *ctx  = &rctx->img;
    int x = 0;
    int y = 0;
    int y_max = MIN(tex->height, ctx->height);
    int x_max = MIN(ctx->width, tex->width);

    for (y = 0; y < y_max; y++) {

        int offset = y * (ctx->width);
        float *r = ctx->r + offset;
        float *g = ctx->g + offset;
        float *b = ctx->b + offset;
        float *a = ctx->a + offset;

        int tex_offset = y * (tex->width);

        float *s_r = tex->r + tex_offset;
        float *s_g = tex->g + tex_offset;
        float *s_b = tex->b + tex_offset;
        float *s_a = tex->a + tex_offset;

        for (x = 0; x < x_max; x++) {

            float alpha = *a;

            //alpha *= .5;

            *r = alpha*(*r) + (1-alpha) *  (*s_r);
            *g = alpha*(*g) + (1-alpha) *  (*s_g);
            *b = alpha*(*b) + (1-alpha) *  (*s_b);
            *a = *s_a;
            //printf("%f\n", *s_r );

            s_r++;
            s_g++;
            s_b++;
            s_a++;
            r++;
            g++;
            b++;
            a++;
        }

    }

}

static inline vec4 get_pixel(Texture *tex, int x, int y)
{
    vec4 color = {0, 0, 0, 0};

    int w = tex->width - 1;
    int h = tex->height - 1;

    x = MIN(MAX(x, 0), w);
    y = MIN(MAX(y, 0), h);

    int index = x + (y * tex->width);

    color.x = tex->r[index];
    color.y = tex->g[index];
    color.z = tex->b[index];
    color.w = tex->a[index];

    return color;
}

static inline float get_alpha_pixel(Texture *tex, int x, int y)
{
    vec4 color = {0, 0, 0, 0};

    int w = tex->width - 1;
    int h = tex->height - 1;

    x = MIN(MAX(x, 0), w);
    y = MIN(MAX(y, 0), h);

    int index = x + (y * tex->width);
    return  tex->a[index];
}

static inline void set_pixel(Texture *ctx, int x, int y, vec4 *color)
{
    if (x > ctx->width  - 1 ||
        y > ctx->height - 1 ) {
        // printf("%d %d out of bounds \n", x,y);
        return;
    }
    int index = x + (y * ctx->width);

    ctx->r[index] = color->x;
    ctx->g[index] = color->y;
    ctx->b[index] = color->z;
    ctx->a[index] = color->w;
}

void grow_texture_slow(Texture *ctx)
{

    int dir[4][2] = {{1,0}, {0, -1}, {-1, 0}, {0,1}};
    //int dir[4][2] = {{1,0}, {-1, 0}};
    vec4 test_pixel;

    int x_p;
    int y_p;
    int step;
    int dir_index, x, y;

    for (dir_index = 0; dir_index < ARRAY_SIZE(dir); dir_index++) {
        int x_d = dir[dir_index][0];
        int y_d = dir[dir_index][1];

        step = 1;
        if (x_d < 0)
            step = -1;

        for (y = 0; y < ctx->height; y++) {

            x_p = 0;
            if (x_d < 0)
                x_p = (ctx->width-1);

            y_p = y;
            if (y_d < 0)
                y_p = (ctx->height-1) - y;

            int pix_index = x_p + (y_p * ctx->width);

            float *r = ctx->r + pix_index;
            float *g = ctx->g + pix_index;
            float *b = ctx->b + pix_index;
            float *a = ctx->a + pix_index;

            for (x = 0; x < ctx->width; x++) {

                test_pixel = get_pixel(ctx, x_p + x_d, y_p + y_d);

                float alpha = 1.0f - *a;
                *r += test_pixel.x * alpha;
                *g += test_pixel.y * alpha;
                *b += test_pixel.z * alpha;

                *a += test_pixel.w * alpha;

                r += step;
                g += step;
                b += step;
                a += step;

                x_p += step;
            }
        }
    }
}

static inline __m128 pix_shift_left(__m128 a, __m128 b)
{
    __m128 l = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4));
    __m128 r = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(b), 12));
    __m128 result = _mm_or_ps(l, r);
    return result;
}

static inline __m128 pix_shift_right(__m128 a, __m128 b)
{
    __m128 l = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(a), 4));
    __m128 r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(b), 12));
    __m128 result = _mm_or_ps(l, r);
    return result;
}

static inline void grow_texture_left_right(Texture *ctx, int right)
{
    __m128 one  = _mm_set1_ps(1);
    __m128 zero = _mm_set1_ps(0);

    int offset = 4;
    if (right)
        offset = -4;
    int x, y;
    for (y = 0; y < ctx->height; y++) {

        int x_p = 0;
        if (right) {
            x_p = ctx->width - 4;
        }

        int pix_index = x_p + (y * ctx->width);
        //printf("%d %d\n", x_p, pix_index);

        float *r = ctx->r + pix_index;
        float *g = ctx->g + pix_index;
        float *b = ctx->b + pix_index;
        float *a = ctx->a + pix_index;

        __m128 r4x = _mm_load_ps(r);
        __m128 g4x = _mm_load_ps(g);
        __m128 b4x = _mm_load_ps(b);
        __m128 a4x = _mm_load_ps(a);

        r+=offset;
        g+=offset;
        b+=offset;
        a+=offset;

        __m128 r_shift;
        __m128 g_shift;
        __m128 b_shift;
        __m128 a_shift;

        for (x = 0; x < ctx->width-4; x+=4) {

            __m128 sr4x = _mm_load_ps(r);
            __m128 sg4x = _mm_load_ps(g);
            __m128 sb4x = _mm_load_ps(b);
            __m128 sa4x = _mm_load_ps(a);

            if (_mm_movemask_epi8(_mm_castps_si128(_mm_cmpeq_ps(a4x, one)))  != 0xFFFF) {

                if(right) {
                    r_shift = pix_shift_right(r4x, sr4x);
                    g_shift = pix_shift_right(g4x, sg4x);
                    b_shift = pix_shift_right(b4x, sb4x);
                    a_shift = pix_shift_right(a4x, sa4x);
                } else {
                    r_shift = pix_shift_left(r4x, sr4x);
                    g_shift = pix_shift_left(g4x, sg4x);
                    b_shift = pix_shift_left(b4x, sb4x);
                    a_shift = pix_shift_left(a4x, sa4x);
                }

                if (_mm_movemask_epi8(_mm_castps_si128(_mm_cmpeq_ps(a_shift, zero))) != 0xFFFF ) {

                    //__m128 alpha = _mm_sub_ps(one, a4x);
                    __m128 alpha = _mm_cmpeq_ps(zero, a4x);

                    __m128 result_r = _mm_or_ps(r4x, _mm_and_ps(alpha, r_shift));
                    __m128 result_g = _mm_or_ps(g4x, _mm_and_ps(alpha, g_shift));
                    __m128 result_b = _mm_or_ps(b4x, _mm_and_ps(alpha, b_shift));
                    __m128 result_a = _mm_or_ps(a4x, _mm_and_ps(alpha, a_shift));

                    _mm_store_ps(r-offset, result_r);
                    _mm_store_ps(g-offset, result_g);
                    _mm_store_ps(b-offset, result_b);
                    _mm_store_ps(a-offset, result_a);
                }
            }

            // _mm_store_ps(r-offset, r_shift);
            // _mm_store_ps(g-offset, g_shift);
            // _mm_store_ps(b-offset, b_shift);
            // _mm_store_ps(a-offset, a_shift);

            r+=offset;
            g+=offset;
            b+=offset;
            a+=offset;

            r4x = sr4x;
            g4x = sg4x;
            b4x = sb4x;
            a4x = sa4x;
        }
    }
}

static inline void grow_texture_down_up(Texture *ctx, int up)
{
    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_set1_ps(0);
    //down and up

    int offset = 1;
    if (up)
        offset = -1;
    int x, y;
    for (y = 0; y < ctx->height-1; y++) {

        int px_y = y;
        if (up)
            px_y =  (ctx->height-1) - y;

        int pix_index = (px_y * ctx->width);

        float *r = ctx->r + pix_index;
        float *g = ctx->g + pix_index;
        float *b = ctx->b + pix_index;
        float *a = ctx->a + pix_index;

        pix_index = ((px_y+offset) * ctx->width);

        float *sr = ctx->r + pix_index;
        float *sg = ctx->g + pix_index;
        float *sb = ctx->b + pix_index;
        float *sa = ctx->a + pix_index;

        for (x = 0; x < ctx->width; x+=4) {
            __m128 a4x = _mm_load_ps(a);
            __m128 sa4x = _mm_load_ps(sa);

            if (_mm_movemask_epi8(_mm_castps_si128(_mm_cmpeq_ps(a4x, one)))   != 0xFFFF ||
                _mm_movemask_epi8(_mm_castps_si128(_mm_cmpeq_ps(sa4x, zero))) != 0xFFFF ) {

                __m128 r4x = _mm_load_ps(r);
                __m128 g4x = _mm_load_ps(g);
                __m128 b4x = _mm_load_ps(b);


                __m128 sr4x = _mm_load_ps(sr);
                __m128 sg4x = _mm_load_ps(sg);
                __m128 sb4x = _mm_load_ps(sb);

                // __m128 alpha = _mm_sub_ps(one, a4x);
                __m128 alpha = _mm_cmpeq_ps(zero, a4x);

                _mm_store_ps(r, _mm_or_ps( r4x, _mm_and_ps( alpha, sr4x)));
                _mm_store_ps(g, _mm_or_ps( g4x, _mm_and_ps( alpha, sg4x)));
                _mm_store_ps(b, _mm_or_ps( b4x, _mm_and_ps( alpha, sb4x)));
                _mm_store_ps(a, _mm_or_ps( a4x, _mm_and_ps( alpha, sa4x)));
            }

            r+=4;
            g+=4;
            b+=4;
            a+=4;
            sr+=4;
            sg+=4;
            sb+=4;
            sa+=4;
        }
    }


}
static const float gauss3x3table[] = {
  1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f,
   1.0f/8.0f, 1.0f/4.0f,  1.0f/8.0f,
  1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f };

static inline vec4 gausse3x3(Texture *ctx, int px, int py)
{
    vec4 average = {};
    int g_index = 0;
    int x,y;
    for (y = py - 1; y <= py + 1; y++) {
        for (x = px - 1; x <= px + 1; x++ ) {
            vec4 pix = get_pixel(ctx, x, y);

            // if (pix.w > 0) {
            //     float inv = 1.0/pix.w;
            //     pix.x*=inv;
            //     pix.y*=inv;
            //     pix.z*-inv;
            // }

            // average.x += pix.x;
            // average.y += pix.y;
            // average.z += pix.z;
            // average.w += pix.w;

            average.x += pix.x * gauss3x3table[g_index];
            average.y += pix.y * gauss3x3table[g_index];
            average.z += pix.z * gauss3x3table[g_index];
            average.w += pix.w * gauss3x3table[g_index];
            g_index++;
            //average.x =1;
        }
    }
    // average.x *= average.w;
    // average.y *= average.w;
    // average.z *= average.w;

    return average;
}

void blur_pixels(Texture *ctx, float *orig_a)
{
    Texture src = {};
    src.width = ctx->width;
    src.height = ctx->width;
    copy_texture(ctx, &src);
    int x,y;
    for (y = 0; y < ctx->height; y++) {

        int pix_index = y * ctx->width;
        float *a  = src.a + pix_index;
        float *oa = orig_a + pix_index;

        for (x = 0; x < ctx->width; x++) {
            if (*oa != *a) {

                //printf("here %d %d\n", x,y);
                vec4 blur = gausse3x3(&src, x, y);

                //vec4 average = {0,0,1,1};
                set_pixel(ctx, x, y, &blur);
            }

            a++;
            oa++;
        }
    }

    free_texture_context(&src);

}

void grow_texture(Texture *ctx)
{
    // int channel_size =  ctx->width * ctx->height * sizeof(float);
    // float *a = (float *)malloc(channel_size);
    // memcpy(a, ctx->a, channel_size);

    grow_texture_down_up(ctx, 0);
    grow_texture_left_right(ctx, 1);
    grow_texture_down_up(ctx, 1);
    grow_texture_left_right(ctx, 0);
    // blur_pixels(ctx, a);
    // free(a);
}

static inline int copy_texture_rect(Texture *src, Texture *dst,
                                      int src_x, int src_y, int dst_x, int dst_y,
                                      int width, int height)
{
    vec4 c;
    int has_pixels = 0;

    for (int y=0; y < height; y++) {
        for (int x=0; x < width; x++) {
            c = get_pixel(src, src_x + x, src_y + y);
            set_pixel(dst, dst_x + x, dst_y + y, &c);
            if (c.w != 0.0f) {
                has_pixels = 1;
            }
        }
    }
    return has_pixels;

    // return 1;
}

static inline int solid_alpha(Texture *tex, int src_x, int src_y, size_t width, size_t height)
{

    size_t w = MIN(tex->width, src_x + width);
    size_t h = MIN(tex->height, src_y + height);

    float *a;

    for (int y=src_y; y < h; y++) {
        size_t index = src_x + (y * tex->width);
        a = tex->a + index;

        for (int x=src_x; x < w; x++) {
            if (*a  == 0.0f) {
                // printf("non soild at %d %d\n", x,y);
                return 0;
            }

            a++;
        }
    }
    return 1;
}

#define GROW_BOARDER 20
#define GROW_BOX 64
#define GROW_SIZE ((GROW_BOARDER * 2) + GROW_BOX)

static inline void replace_texture(Texture *src, Texture *dst)
{
    free_texture_context(dst);
    dst->mem = src->mem;
    dst->mem_size = src->mem_size;
    dst->width = src->width;
    dst->height = src->height;
    dst->r = src->r;
    dst->g = src->g;
    dst->b = src->b;
    dst->a = src->a;
}


void grow_texture_new(Texture *ctx, Texture *dst, const Rect *clip)
{
    Texture tmp_tex;
    tmp_tex.mem = NULL;

#if 0
    setup_texture_context(&tmp_tex, GROW_SIZE, GROW_SIZE);
#else
    // use stack memory for temp texture
    float texture_memory[GROW_SIZE * GROW_SIZE * 4] __attribute__((aligned(16)));

    size_t channel_size = GROW_SIZE * GROW_SIZE;

    tmp_tex.width = GROW_SIZE;
    tmp_tex.height = GROW_SIZE;

    tmp_tex.r = &texture_memory[0];
    tmp_tex.g = tmp_tex.r + channel_size;
    tmp_tex.b = tmp_tex.g + channel_size;
    tmp_tex.a = tmp_tex.b + channel_size;

#endif

    // printf("%dx%d\n", ctx->width, ctx->height);
    // printf("%dx%d\n", tmp_tex.width, tmp_tex.height);
    // printf("%dx%d\n", dst->width, dst->height);

    int width  =  clip->max.x - clip->min.x + 1;
    int height =  clip->max.y - clip->min.y + 1;

    // copy_texture_rect2(ctx, dst, clip, clip->min.x, clip->min.y);

    int has_pixels = copy_texture_rect(ctx, dst,
                                       clip->min.x, clip->min.y, clip->min.x, clip->min.y,
                                       width, height);

    if (!has_pixels)
        return;


    int x_steps = width / GROW_BOX;
    int y_steps = width / GROW_BOX;

    int src_y = clip->min.y;

    for (int y =0; y <= y_steps; y++) {
        int src_x = clip->min.x;
        for (int x=0; x <= x_steps; x++) {

            // Texture *a_tex = &tmp1_tex;

            has_pixels = copy_texture_rect(ctx, &tmp_tex,
                                         src_x - GROW_BOARDER, src_y - GROW_BOARDER,
                                         0, 0, GROW_SIZE, GROW_SIZE);

            if (has_pixels) {
                if(!solid_alpha( &tmp_tex, GROW_BOARDER, GROW_BOARDER, GROW_BOX, GROW_BOX)) {

                    for (int i = 0; i < GROW_BOARDER-4; i++) {
                        grow_texture_down_up(&tmp_tex, 0);
                        if (solid_alpha(&tmp_tex, GROW_BOARDER, GROW_BOARDER, GROW_BOX, GROW_BOX))
                            break;

                        grow_texture_left_right(&tmp_tex, 1);
                        if (solid_alpha(&tmp_tex, GROW_BOARDER, GROW_BOARDER, GROW_BOX, GROW_BOX))
                            break;
                        grow_texture_down_up(&tmp_tex, 1);
                        if (solid_alpha(&tmp_tex, GROW_BOARDER, GROW_BOARDER, GROW_BOX, GROW_BOX))
                            break;
                        grow_texture_left_right(&tmp_tex, 0);
                        if (solid_alpha( &tmp_tex, GROW_BOARDER, GROW_BOARDER, GROW_BOX, GROW_BOX))
                            break;

                    }

                    int copy_width  = MIN(GROW_BOX, clip->max.x - src_x + 1);
                    int copy_height = MIN(GROW_BOX, clip->max.y - src_y + 1);

                    copy_texture_rect(&tmp_tex, dst,
                                      GROW_BOARDER, GROW_BOARDER,
                                      src_x, src_y, copy_width, copy_height);
                }
            }
            src_x += GROW_BOX;

        }
        src_y += GROW_BOX;
    }

    // replace_texture(&dst_tex, ctx);
    // free_texture_context(&tmp1_tex);
}

static inline vec4 get_tex_color(Texture *tex, vec2 uv)
{
    int w = tex->width - 1;
    int h = tex->height - 1;

    int x = (uv.x) * w;
    int y = (1 - uv.y) * h;

    return get_pixel(tex, x, y);
}

static inline vec4 get_tex_color_linear(Texture *tex, vec2 uv)
{
    int width = tex->width - 1;
    int height = tex->height - 1;

    float x = (uv.x) * (float)width;
    float y = (1.0f - uv.y) * (float)height;

    int px = (int)(x); //floor
    int py = (int)(y); //floor

    vec4 c[4];
    c[0] = get_pixel(tex, px + 0, py + 0);
    c[1] = get_pixel(tex, px + 1, py + 0);
    c[2] = get_pixel(tex, px + 0, py + 1);
    c[3] = get_pixel(tex, px + 1, py + 1);

    float fx = x - (float)px;
    float fy = y - (float)py;
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;

    float w[4];

    w[0] = fx1 * fy1;
    w[1] = fx  * fy1;
    w[2] = fx1 * fy;
    w[3] = fx  * fy;

    vec4 color;

#if 1
    color.sse = _mm_add_ps(_mm_mul_ps(c[0].sse, _mm_set1_ps(w[0])),
                _mm_add_ps(_mm_mul_ps(c[1].sse, _mm_set1_ps(w[1])),
                _mm_add_ps(_mm_mul_ps(c[2].sse, _mm_set1_ps(w[2])),
                           _mm_mul_ps(c[3].sse, _mm_set1_ps(w[3])))));
#else
    color.x = (c[0].x * w[0]) + (c[1].x * w[1]) + (c[2].x * w[2]) + (c[3].x * w[3]);
    color.y = (c[0].y * w[0]) + (c[1].y * w[1]) + (c[2].y * w[2]) + (c[3].y * w[3]);
    color.z = (c[0].z * w[0]) + (c[1].z * w[1]) + (c[2].z * w[2]) + (c[3].z * w[3]);
    color.w = (c[0].w * w[0]) + (c[1].w * w[1]) + (c[2].w * w[2]) + (c[3].w * w[3]);
#endif

    return color;
}

void resample_texture_half(Texture *src, Texture *dst)
{
    printf("fast half\n");
    for (int y_pixel = 0; y_pixel < dst->height; y_pixel++) {
        int y = y_pixel * 2;
        for (int x_pixel = 0; x_pixel < dst->width; x_pixel++) {
            int x = x_pixel *2;

            vec4 c1 = get_pixel(src, x, y);
            vec4 c2 = get_pixel(src, x+1, y);
            vec4 c3 = get_pixel(src, x, y+1);
            vec4 c4 = get_pixel(src, x+1, y+1);

            vec4 c;
            // c = div_vec4_f(add_vec4(c1, add_vec4(c2, add_vec4(c3, c4))), 4.0f);

            c.sse = _mm_mul_ps(_mm_add_ps(c1.sse, _mm_add_ps(c2.sse, _mm_add_ps(c3.sse, c4.sse))), _mm_set1_ps(0.25f));

            set_pixel(dst, x_pixel, y_pixel, &c);

        }
    }
}

#define _MM_SHUFFLE_R(a,b,c,d) _MM_SHUFFLE(d,c,b,a)

static inline __m128 get_4x8_half(float *p)
{

    __m128 a = _mm_load_ps(p);
    __m128 b = _mm_load_ps(p + 4);

    __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF,0,0xFFFFFFFF,0));

    a = _mm_add_ps(a, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)));
    a = _mm_and_ps(a, mask);
    a = _mm_shuffle_epi32(a, _MM_SHUFFLE_R(0,2,1,3));

    b = _mm_add_ps(b, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(b), 4)));
    b = _mm_and_ps(b, mask);
    b = _mm_shuffle_epi32(b, _MM_SHUFFLE_R(1,3,0,2));

    return _mm_or_ps(a,b);
}

void resample_texture_half_sse(Texture *src, Texture *dst)
{
    int sw = src->width - 1;
    int sh = src->height - 1;

    // printf("fast half sse\n");

    float *src_channels[4];
    float *dst_channels[4];

    src_channels[0] = src->r;
    src_channels[1] = src->g;
    src_channels[2] = src->b;
    src_channels[3] = src->a;

    dst_channels[0] = dst->r;
    dst_channels[1] = dst->g;
    dst_channels[2] = dst->b;
    dst_channels[3] = dst->a;

    for (int i = 0; i < 4; i++) {

        float *src_chan = src_channels[i];
        float *dst_chan = dst_channels[i];

        for (int y_pixel = 0; y_pixel < src->height; y_pixel+=2) {
            for (int x_pixel = 0; x_pixel < src->width; x_pixel+=8) {
                __m128 a = get_4x8_half(src_chan);
                __m128 b = get_4x8_half(src_chan + src->width);
                __m128 c = _mm_mul_ps(_mm_add_ps(a, b), _mm_set1_ps(0.25f));

                _mm_store_ps(dst_chan, c);

                src_chan += 8;
                dst_chan += 4;
            }
            src_chan += src->width;
        }
    }

}

void resample_texture(Texture *src, Texture *dst)
{
    vec2 uv;

    if (src->width/2 == dst->width && src->height/2 == dst->height) {

        if (src->width % 8 == 0) {
            resample_texture_half_sse(src, dst);
            return;
        }

        resample_texture_half(src, dst);
        return;
    }


    for (int y_pixel = 0; y_pixel < dst->height; y_pixel++) {

        for (int x_pixel = 0; x_pixel < dst->width; x_pixel++) {
            vec4 r;
            uv.y = (y_pixel + 0.5f) / (float)dst->height;
            uv.x = (x_pixel + 0.5f) / (float)dst->width;
            vec4 c  =  get_tex_color_linear(src, uv);
            set_pixel(dst, x_pixel, (dst->height-1) - y_pixel, &c);

        }
    }
}

typedef struct {
    __m128 x;
    __m128 y;
    __m128 z;
    __m128 w;
} m128vec4;

static inline float halfedgef(const vec4 *a, const vec4 *b, const vec4 *c)
{
    float result = ((b->x - a->x) * (c->y - a->y)) - ((b->y - a->y) * (c->x - a->x));
    return result;
}

static inline __m128 halfedge_x4(const m128vec4 *a, const m128vec4 *b, const m128vec4 *c)
{
    //float result = ((b->x - a->x) * (c->y - a->y)) -
    //               ((b->y - a->y) * (c->x - a->x));

    __m128 result = _mm_sub_ps(
                        _mm_mul_ps(_mm_sub_ps(b->x, a->x), _mm_sub_ps(c->y, a->y)),
                        _mm_mul_ps(_mm_sub_ps(b->y, a->y), _mm_sub_ps(c->x, a->x)));
    return result;
}

static inline int halfedgei(const vec2i *a, const vec2i *b, const vec2i *c)
{
    int result = ((b->x - a->x) * (c->y - a->y)) - ((b->y - a->y) * (c->x - a->x));
    return result;
}


static inline float bary_blend(float v0, float v1, float v2, float w0, float w1, float w2)
{
    float result;
    result = (v0 * w0) + (v1 * w1) + (v2 * w2);
    return result;
}

static inline vec4 blend_vec4(const vec4 *v0, const vec4 *v1, const vec4 *v2, float w0, float w1, float w2)
{
    vec4 result;
    result.x = bary_blend(v0->x, v1->x, v2->x, w0, w1, w2);
    result.y = bary_blend(v0->y, v1->y, v2->y, w0, w1, w2);
    result.z = bary_blend(v0->z, v1->z, v2->z, w0, w1, w2);
    result.w = bary_blend(v0->w, v1->w, v2->w, w0, w1, w2);
    return result;
}

static inline vec3 blend_vec3(const vec3 *v0, const vec3 *v1, const vec3 *v2, float w0, float w1, float w2)
{
    vec3 result;
    result.x = bary_blend(v0->x, v1->x, v2->x, w0, w1, w2);
    result.y = bary_blend(v0->y, v1->y, v2->y, w0, w1, w2);
    result.z = bary_blend(v0->z, v1->z, v2->z, w0, w1, w2);
    return result;
}

static inline vec2 blend_vec2(const vec2 *v0, const vec2 *v1, const vec2 *v2, float w0, float w1, float w2)
{
    vec2 result;
    result.x = bary_blend(v0->x, v1->x, v2->x, w0, w1, w2);
    result.y = bary_blend(v0->y, v1->y, v2->y, w0, w1, w2);
    return result;
}

static inline __m128 bary_blend_m128(__m128 v0, __m128 v1, __m128 v2,
                              __m128 w0, __m128 w1, __m128 w2)
{
    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(v0, w0), _mm_mul_ps(v1, w1)), _mm_mul_ps(v2, w2));
}

static inline int rect_intersects(Rect *a, Rect *b)
{
    if (a->min.x <= b->max.x && a->max.x >= b->min.x &&
        a->min.y <= b->max.y && a->max.y >= b->min.y)
        return 1;
    return 0;
}


static inline void load_mesh_data(Triangle* p, const MeshIndex indices[3], const MeshData *mesh, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        p->verts[i] = *((vec4*)&mesh->vertices[indices[i].v * 4]);

        if (mesh->coarse_levels)
            p->coarse_levels[i] = mesh->coarse_levels[indices[i].v];

        if (mesh->normals) {
            p->has_normals = 1;
            p->normals[i] = *((vec3*)&mesh->normals[indices[i].n * 3]);
        }

        if (mesh->uvs) {
            p->has_uvs = 1;
            p->uvs[i] = *((vec2*)&mesh->uvs[indices[i].uv * 2]);
        }
    }

}

static inline void perspective_divide(Triangle* p)
{
    float w[3] = {p->verts[0].w, p->verts[1].w, p->verts[2].w};
    int i;
    for (i =0; i < 3; i++) {
        p->verts[i].x /= w[i];
        p->verts[i].y /= w[i];
        p->verts[i].z /= w[i];
    }
}

static inline void perspective_correction(Triangle* p)
{
    float w[3] = {p->verts[0].w, p->verts[1].w, p->verts[2].w};

    int i;
    for (i =0; i < 3; i++) {
        float inv_w = 1.0f / w[i];
        if (p->has_normals) {
            p->normals[i].x *= inv_w;
            p->normals[i].y *= inv_w;
            p->normals[i].z *= inv_w;
        }

        if (p->has_uvs) {
            p->uvs[i].x *= inv_w;
            p->uvs[i].y *= inv_w;
        }

        p->c1[i].x *= inv_w;
        p->c1[i].y *= inv_w;
        p->c1[i].z *= inv_w;

        p->c2[i].x *= inv_w;
        p->c2[i].y *= inv_w;
        p->c2[i].z *= inv_w;

    }
}

void render_pixel(RenderContext *ctx, const vec4 *point,
                  float w0, float w1, float w2, const Triangle* poly, Texture *tex)
{

    int x = point->x;
    int y = point->y;

    int width = ctx->width;

    int offset = x + (y * (width));


    float *r  = ctx->img.r  + offset;
    float *g  = ctx->img.g  + offset;
    float *b  = ctx->img.b  + offset;
    float *a  = ctx->img.a  + offset;
    float *z  = ctx->z  + offset;

    vec4 c1 = {0.0f, 0.0f, 0.0f};
    vec4 c2 = {0.0f, 0.0f, 0.0f};

    vec3 normal = {1.0f, 1.0f, 1.0f};
    vec2 uv = {0.0f, 0.0f};

    vec4 tc = {0, 0, 0, 0};

    float one_over_z = 1.0;
    float correct_z = 1.0;

    if (ctx->perspective_correct) {
        one_over_z = bary_blend(1.0f / poly->verts[0].w,
                                1.0f / poly->verts[1].w,
                                1.0f / poly->verts[2].w,
                                w0, w1, w2);
        correct_z = 1.0f/one_over_z;
    }

    c1 = blend_vec4(&poly->c1[0], &poly->c1[1], &poly->c1[2], w0, w1, w2);
    // if (ctx->perspective_correct) {
    //     c1.x *= correct_z;
    //     c1.y *= correct_z;
    //     c1.z *= correct_z;
    // }

    c2 = blend_vec4(&poly->c2[0], &poly->c2[1], &poly->c2[2], w0, w1, w2);
    if (!ctx->uvspace && ctx->perspective_correct) {
        c2.x *= correct_z;
        c2.y *= correct_z;
        c2.z *= correct_z;
    }

    if (ctx->projection && tex) {
        vec2 pos = {0,0};
        pos.x = c2.x;
        pos.y = c2.y;
        pos.x = (pos.x / (float)(tex->width-1));
        pos.y = 1.0f - (pos.y / (float)(tex->height-1));

        tc = get_tex_color_linear(tex, pos);

        *r = tc.x;
        *g = tc.y;
        *b = tc.z;
        *a = 1.0f;
        return;
    }

    if (poly->has_normals) {
        normal = blend_vec3(&poly->normals[0], &poly->normals[1], &poly->normals[2], w0, w1, w2);
        if (!ctx->uvspace && ctx->perspective_correct) {
            normal.x *= correct_z;
            normal.y *= correct_z;
            normal.z *= correct_z;
        }

    }

    if (poly->has_uvs) {

        uv = blend_vec2(&poly->uvs[0], &poly->uvs[1], &poly->uvs[2], w0, w1, w2);
        if (!ctx->uvspace && ctx->perspective_correct) {
            uv.x *= correct_z;
            uv.y *= correct_z;
        }
        if (tex)
            tc = get_tex_color_linear(tex, uv);
    }

    int m = ctx->checker_size;
    float checker = (fmod(uv.x * m, 1.0) > 0.5) ^ (fmod(uv.y * m, 1.0) < 0.5);
    //float checker = (fmod(x/(float)ctx->width * m, 1.0) > 0.5) ^ (fmod(y/(float)ctx->height * m, 1.0) < 0.5);
    checker = checker * 0.5f + 0.5f;

    vec3 light_dir = {0, 0, 1};
    //a.b = a.x * b.x + a.y * b.y
    float dot = (light_dir.x * normal.x) +
                (light_dir.y * normal.y) +
                (light_dir.z * normal.z);

    //float l =
    // __m128 length = _mm_sqrt_ps(_mm_mul_ps(dot,dot));
    // __m128 light_amt = _mm_add_ps(_mm_mul_ps(length, _mm_set1_ps(0.9f)), _mm_set1_ps(0.1f) );

    *r = dot * normal.x * checker + (.1 *w0) + .1;
    *g = dot * normal.y * checker + (.1 *w1) + .1;
    *b = dot * normal.z * checker + (.1 *w2) + .1;


    float tc_a = 1.0f - tc.w;

    *r = *r * tc_a + tc.x;
    *g = *g * tc_a + tc.y;
    *b = *b * tc_a + tc.z;

    // *r = tc.x;
    // *g = tc.y;
    // *b = tc.z;

    if (!tex) {
        *r = c1.x;
        *g = c1.y;
        *b = c1.z;
    }

    *a = 1.0f;
}

static inline void uv_to_screen_space(const RenderContext *ctx, Triangle *p, Texture *tex)
{    if (p->has_uvs) {


      p->c2[0] = p->verts[0];
      p->c2[1] = p->verts[1];
      p->c2[2] = p->verts[2];

      int w = ctx->width - 1;
      int h = ctx->height - 1;

      p->verts[0].x =  p->uvs[0].x * w;
      p->verts[0].y =  (1 -p->uvs[0].y) * h;
      p->verts[0].z = -1.0f;
      //p->verts[0].w = 1.0f;

      p->verts[1].x =  p->uvs[1].x * w;
      p->verts[1].y =  (1-p->uvs[1].y) * h;
      p->verts[1].z = -1.0f;
      //p->verts[1].w = 1.0f;

      p->verts[2].x =  p->uvs[2].x * w;
      p->verts[2].y =  (1-p->uvs[2].y) * h;
      p->verts[2].z = -1.0f;
      //p->verts[2].w = 1.0f;

    }

    //printf("%f %f\n", p->verts[2].x, p->verts[2].y);
}

inline void flip_triangle(Triangle* p)
{
    Triangle tmp = *p;
    int i;
    for (i =0; i < 3; i++) {
        p->verts[i] = tmp.verts[2-i];
        p->normals[i] = tmp.normals[2-i];
        p->uvs[i] = tmp.uvs[2-i];
    }
}

inline int top_or_left_edge(vec4 *a, vec4 *b)
{
    int   dy = b->y - a->y + 0.5f;
    float dx = b->x - a->x;
    if ((dy > 0) || (dy == 0 && dx < 0))
         return 1;
    return 0;
}

static inline m128vec4 set1_m128vec4(vec4 *a)
{
    m128vec4 v;
    v.x = _mm_set1_ps(a->x);
    v.y = _mm_set1_ps(a->y);
    v.z = _mm_set1_ps(a->z);
    v.w = _mm_set1_ps(a->w);
    return v;
}

static inline vec4 m128vec4_get_vec4(m128vec4 *a, int i) {
    vec4 v;
    v.x = a->x[i];
    v.y = a->y[i];
    v.z = a->z[i];
    v.w = a->w[i];
    return v;
}

void draw_triangle(RenderContext *ctx, const Triangle *tri, const Rect *clip, const Texture *tex)
{
    int width = ctx->width;
    int height = ctx->height;

    vec4 *v0 = &tri->verts[0];
    vec4 *v1 = &tri->verts[1];
    vec4 *v2 = &tri->verts[2];

    float area = halfedgef(v0, v1, v2);
    if (area < 0.0f) {
        return;
    }

    // Compute triangle bounding box
    float min_xf = MIN3(v0->x, v1->x, v2->x);
    float min_yf = MIN3(v0->y, v1->y, v2->y);
    float max_xf = MAX3(v0->x, v1->x, v2->x);
    float max_yf = MAX3(v0->y, v1->y, v2->y);

    Rect bbox;
    bbox.min.x = min_xf;
    bbox.min.y = min_yf;

    bbox.max.x = max_xf;
    bbox.max.y = max_yf;

    if (!rect_intersects(clip, &bbox)) {
        return;
    }

    int min_x = floorf(MAX3(min_xf, clip->min.x, 0));
    int min_y = floorf(MAX3(min_yf, clip->min.y, 0));
    int max_x = ceilf( MIN3(max_xf, clip->max.x, width - 1));
    int max_y = ceilf( MIN3(max_yf, clip->max.y, height - 1));

    // float w0_bias = top_or_left_edge(v1, v2) ? 0 : -1;
    // float w1_bias = top_or_left_edge(v2, v0) ? 0 : -1;
    // float w2_bias = top_or_left_edge(v0, v1) ? 0 : -1;

    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_set1_ps(0);
    __m128 four = _mm_set1_ps(4);
    __m128 area4x_inv = _mm_set1_ps(1.0f / area);

    m128vec4 v0x4 = set1_m128vec4(v0);
    m128vec4 v1x4 = set1_m128vec4(v1);
    m128vec4 v2x4 = set1_m128vec4(v2);

    vec4 p;
    m128vec4 px4;
    int i,x,y;
    for (y = min_y; y <= max_y; y+=2) {

        px4.x = _mm_add_ps(_mm_set1_ps(min_x), _mm_setr_ps(0,1,0,1));
        px4.y = _mm_add_ps(_mm_set1_ps(y), _mm_setr_ps(0,0,1,1));

        px4.y = _mm_min_ps(px4.y , _mm_set1_ps(clip->max.y));
        //p.y = y;

        for (x = min_x; x <= max_x; x+=2) {
            __m128 w0x4 = halfedge_x4(&v1x4, &v2x4, &px4);
            __m128 w1x4 = halfedge_x4(&v2x4, &v0x4, &px4);
            __m128 w2x4 = halfedge_x4(&v0x4, &v1x4, &px4);

            __m128 inside4x = _mm_and_ps(_mm_cmpge_ps(w0x4, zero),
                              _mm_and_ps(_mm_cmpge_ps(w1x4, zero),
                                         _mm_cmpge_ps(w2x4, zero)));
            //if all are not zero at least one is inside the triangle
            if (_mm_movemask_epi8(_mm_castps_si128(_mm_cmpeq_ps(inside4x, zero))) != 0xFFFF) {

                w0x4 = _mm_mul_ps(w0x4, area4x_inv);
                w1x4 = _mm_mul_ps(w1x4, area4x_inv);
                w2x4 = _mm_mul_ps(w2x4, area4x_inv);

                // bary blend z values
                px4.z = bary_blend_m128(v0x4.z, v1x4.z, v2x4.z,
                                          w0x4,   w1x4,   w2x4);

                for (i =0; i < 4; i++) {
                    float inside = inside4x[i];
                    p = m128vec4_get_vec4(&px4, i);
                    //p.x += i;
                    if (inside) {
                        float w0 = w0x4[i];
                        float w1 = w1x4[i];
                        float w2 = w2x4[i];

                        int pixel_offset = (int)p.x + ((int)p.y * width);

                        // if passes depth test render pixel
                        if (p.z < ctx->z[pixel_offset]) {
                            //printf("%f\n", p.z);
                            ctx->z[pixel_offset] = p.z;
                            render_pixel(ctx, &p, w0, w1, w2, tri, tex);
                        }
                    }
                }
            }
            px4.x = _mm_add_ps(px4.x, _mm_set1_ps(2));
            px4.x = _mm_min_ps(px4.x , _mm_set1_ps(clip->max.x));
        }
    }
}

void draw_tri_line(RenderContext *ctx, Edge* edge, float width, const Rect *clip, const Texture *tex)
{
    Triangle tri0 = {};
    Triangle tri1 = {};

    float z_offset = -0.01;

    //z_offset = 1;

    width /= 2.0f;

    vec2 v;

    vec4 p0 =  edge->verts[0];
    vec4 p1 =  edge->verts[1];

    v.x =  p1.x - p0.x;
    v.y =  p1.y - p0.y;

    //normalize;
    float inv_len = 1.0f / sqrtf(v.x * v.x + v.y * v.y);
    v.x *= inv_len;
    v.y *= inv_len;

    float scale = width;
    //cw
    vec2 right =  {-v.y * scale,  v.x * scale};
    //ccw
    vec2 left = { v.y * scale, -v.x * scale};

    //bottom
    vec4 v0 =  edge->verts[0];
    v0.x += left.x;
    v0.y += left.y;

    vec4 v1 =  edge->verts[0];
    v1.x += right.x;
    v1.y += right.y;

    //top
    vec4 v2 = edge->verts[1];
    v2.x += left.x;
    v2.y += left.y;

    vec4 v3 = edge->verts[1];
    v3.x += right.x;
    v3.y += right.y;

    v0.z += z_offset;
    v1.z += z_offset;
    v2.z += z_offset;
    v3.z += z_offset;

    tri0.verts[0] = v0;
    tri0.verts[1] = v2;
    tri0.verts[2] = v1;

    tri1.verts[0] = v2;
    tri1.verts[1] = v3;
    tri1.verts[2] = v1;

    tri0.has_normals = edge->has_normals;
    tri1.has_normals = edge->has_normals;

    tri0.has_uvs = edge->has_uvs;
    tri1.has_uvs = edge->has_uvs;

    tri0.normals[0] = edge->normals[0];
    tri0.normals[1] = edge->normals[1];
    tri0.normals[2] = edge->normals[0];

    tri0.uvs[0] = edge->uvs[0];
    tri0.uvs[1] = edge->uvs[1];
    tri0.uvs[2] = edge->uvs[0];

    tri0.c1[0] = edge->c1[0];
    tri0.c1[1] = edge->c1[1];
    tri0.c1[2] = edge->c1[0];

    tri0.c2[0] = edge->c2[0];
    tri0.c2[1] = edge->c2[1];
    tri0.c2[2] = edge->c2[0];

    tri1.normals[0] = edge->normals[1];
    tri1.normals[1] = edge->normals[1];
    tri1.normals[2] = edge->normals[0];

    tri1.c1[0] = edge->c1[1];
    tri1.c1[1] = edge->c1[1];
    tri1.c1[2] = edge->c1[0];

    tri1.c2[0] = edge->c2[1];
    tri1.c2[1] = edge->c2[1];
    tri1.c2[2] = edge->c2[0];

    tri1.uvs[0] = edge->uvs[1];
    tri1.uvs[1] = edge->uvs[1];
    tri1.uvs[2] = edge->uvs[0];

    draw_triangle(ctx, &tri0, clip, tex);
    draw_triangle(ctx, &tri1, clip, tex);
}

void set_edge_color(Edge *edge, vec4* color)
{
    edge->c1[0].x = color->x;
    edge->c1[0].y = color->y;
    edge->c1[0].z = color->z;
    edge->c1[0].w = color->w;

    edge->c1[1].x = color->x;
    edge->c1[1].y = color->y;
    edge->c1[1].z = color->z;
    edge->c1[1].w = color->w;
}

void draw_tri_edges(RenderContext *ctx, const MeshIndex *indices, const Triangle *tri, const Rect *clip, const Texture *tex)
{
    vec4 *v0 = &tri->verts[0];
    vec4 *v1 = &tri->verts[1];
    vec4 *v2 = &tri->verts[2];

    float line_width = 1;

    vec4 diagonal_c = {0,0,1,1};
    vec4 coarse_edge_c = {1,1,1,1};
    vec4 edge_c = {0.25, 0.25, 0.25, 0.25};

    Edge edge;

    edge.has_normals = tri->has_normals;
    edge.has_uvs = tri->has_uvs;

    edge.verts[0] = tri->verts[0];
    edge.verts[1] = tri->verts[1];

    edge.normals[0] = tri->normals[0];
    edge.normals[1] = tri->normals[1];

    edge.c2[0] = tri->c2[0];
    edge.c2[1] = tri->c2[1];

    edge.uvs[0] = tri->uvs[0];
    edge.uvs[1] = tri->uvs[1];

    set_edge_color(&edge, &edge_c);
    if (tri->coarse_levels[0] == 0 && tri->coarse_levels[1] == 0)
        set_edge_color(&edge, &coarse_edge_c);

    if (!indices[0].flags)
        draw_tri_line(ctx, &edge, line_width, clip, NULL);

    edge.verts[0] = tri->verts[1];
    edge.verts[1] = tri->verts[2];

    edge.normals[0] = tri->normals[1];
    edge.normals[1] = tri->normals[2];

    edge.c2[0] = tri->c2[1];
    edge.c2[1] = tri->c2[2];

    edge.uvs[0] = tri->uvs[1];
    edge.uvs[1] = tri->uvs[2];

    set_edge_color(&edge, &edge_c);
    if (tri->coarse_levels[1] == 0 && tri->coarse_levels[2] == 0)
        set_edge_color(&edge, &coarse_edge_c);

    if (!indices[1].flags)
        draw_tri_line(ctx, &edge, line_width, clip, NULL);

    edge.verts[0] = tri->verts[2];
    edge.verts[1] = tri->verts[0];

    edge.normals[0] = tri->normals[2];
    edge.normals[1] = tri->normals[0];

    edge.c2[0] = tri->c2[2];
    edge.c2[1] = tri->c2[0];

    edge.uvs[0] = tri->uvs[2];
    edge.uvs[1] = tri->uvs[0];

    set_edge_color(&edge, &edge_c);
    if (tri->coarse_levels[2] == 0 && tri->coarse_levels[0] == 0)
        set_edge_color(&edge, &coarse_edge_c);

    if (!indices[2].flags)
        draw_tri_line(ctx, &edge, line_width, clip, NULL);

    return;
}

void draw_triangle_float(RenderContext *ctx, const MeshIndex indices[3], const MeshData *mesh, const Rect *clip, const Texture *tex)
{
    Triangle tri = {};
    load_mesh_data(&tri, indices, mesh, 3);

    //ctx->perspective_correct = 1;
    //perspective_divide(&tri);

    if (ctx->uvspace || ctx->projection)
        uv_to_screen_space(ctx, &tri, tex);

    if (!ctx->uvspace && ctx->perspective_correct)
       perspective_correction(&tri);

    draw_triangle(ctx, &tri, clip, tex);
    if (ctx->wireframe)
        draw_tri_edges(ctx, indices, &tri, clip, NULL);

}

void draw_edge(RenderContext *ctx, const int vert_indices[2], const MeshData *mesh, const Rect *clip, const Texture *tex)
{
    //void draw_tri_line(RenderContext *ctx, Edge* edge, float width, const Rect *clip, const Texture *tex)

    Edge edge = {};
    float w;

    edge.verts[0] = *((vec4*)&mesh->vertices[vert_indices[0] * 4]);
    edge.verts[1] = *((vec4*)&mesh->vertices[vert_indices[1] * 4]);

    w = edge.verts[0].w;
    edge.verts[0].x /= w;
    edge.verts[0].y /= w;
    edge.verts[0].z /= w;

    w = edge.verts[1].w;
    edge.verts[1].x /= w;
    edge.verts[1].y /= w;
    edge.verts[1].z /= w;

    draw_tri_line(ctx, &edge, 1, clip, tex);

}

void simple_line_draw(RenderContext *ctx, vec4 *a, vec4 *b, vec4 *c, const Rect *clip)
{
    Edge edge = {};
    edge.verts[0] = *a;
    edge.verts[1] = *b;

    //printf("bbox: x: %f y: %f z: %f w: %f\n",  bbox[0].x,  bbox[0].y,  bbox[0].z,  bbox[0].w);
    edge.c1[0] = *c;
    edge.c1[1] = *c;

    draw_tri_line(ctx, &edge, 1, clip, NULL);
}

void draw_bbox(RenderContext *ctx, vec4 *bbox, const Rect *clip)
{

    vec4 c = {1.0, 1.0, 1.0, 1.0};
    simple_line_draw(ctx, &bbox[0], &bbox[1], &c, clip);
    simple_line_draw(ctx, &bbox[0], &bbox[2], &c, clip);
    simple_line_draw(ctx, &bbox[0], &bbox[4], &c, clip);
    simple_line_draw(ctx, &bbox[1], &bbox[3], &c, clip);
    simple_line_draw(ctx, &bbox[1], &bbox[5], &c, clip);

    simple_line_draw(ctx, &bbox[2], &bbox[3], &c, clip);
    simple_line_draw(ctx, &bbox[2], &bbox[6], &c, clip);

    simple_line_draw(ctx, &bbox[3], &bbox[7], &c, clip);

    simple_line_draw(ctx, &bbox[4], &bbox[6], &c, clip);
    simple_line_draw(ctx, &bbox[4], &bbox[5], &c, clip);

    simple_line_draw(ctx, &bbox[5], &bbox[7], &c, clip);

    simple_line_draw(ctx, &bbox[6], &bbox[7], &c, clip);
}

static inline vec3 sub_vec3(const vec3 *a, const vec3 *b)
{
    vec3 r;
    r.x = a->x - b->x;
    r.y = a->y - b->y;
    r.z = a->z - b->z;
    return r;
}

static inline float dot_vec3(const vec3 *a, const vec3 *b)
{
	return a->x * b->x + a->y * b->y + a->z * b->z;
}

static inline float len_squared_vec3(vec3 *a, vec3 *b)
{
    vec3 r = sub_vec3(b, a);
    return dot_vec3(&r, &r);
}


void draw_quad(RenderContext *ctx, const MeshIndex indices[4], const MeshData *mesh, const Rect *clip, const Texture *tex)
{
    float d1, d2;
    MeshIndex p[3];

    vec3 *va = (vec3*)&mesh->objspace_vertices[indices[0].v * 4];
    vec3 *vb = (vec3*)&mesh->objspace_vertices[indices[1].v * 4];
    vec3 *vc = (vec3*)&mesh->objspace_vertices[indices[2].v * 4];
    vec3 *vd = (vec3*)&mesh->objspace_vertices[indices[3].v * 4];

    // calculate shortest diagonal
	d1 = len_squared_vec3(va, vc);
    d2 = len_squared_vec3(vb, vd);

    int dag = 1;

    if (d1 < d2) {
        //d1 is shortest, diagonal is va -> bc
        p[0] = indices[0];
        p[1] = indices[1];
        p[2] = indices[2];
        p[0].flags = 0;
        p[1].flags = 0;
        p[2].flags = dag;
        draw_triangle_float(ctx, p, mesh, clip, tex);
        p[0] = indices[2];
        p[1] = indices[3];
        p[2] = indices[0];
        p[0].flags = 0;
        p[1].flags = 0;
        p[2].flags = dag;
        draw_triangle_float(ctx, p, mesh, clip, tex);
    } else {
        //d2 is shortest, diagonal is vb -> vd
        p[0] = indices[0];
        p[1] = indices[1];
        p[2] = indices[3];

        p[0].flags = 0;
        p[1].flags = dag;
        p[2].flags = 0;
        draw_triangle_float(ctx, p, mesh, clip, tex);
        p[0] = indices[1];
        p[1] = indices[2];
        p[2] = indices[3];
        p[0].flags = 0;
        p[1].flags = 0;
        p[2].flags = dag;
        draw_triangle_float(ctx, p, mesh, clip, tex);
    }
}

static inline __m128i mull_epi32(__m128i a, __m128i b)
{
    __m128i tmp1 = _mm_mul_epu32(a,b); /* mul 2,0*/
    __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
}

static int stepXSize = 4;
static int stepYSize = 1;

static int prec = 4;


static inline __m128 blend_ps(__m128 a, __m128 b, __m128 mask)
{
    __m128 b_masked = _mm_and_ps(mask, b); // Where mask is 1, output b.
    __m128 a_masked= _mm_andnot_ps(mask, a); // Where mask is 0, output a.
    return _mm_or_ps(a_masked, b_masked);

}


inline __m128 modf_ps(__m128 x, __m128 mod)
{
    __m128 ints = _mm_div_ps(x, mod);
    __m128 integerpart = _mm_cvtepi32_ps(_mm_cvttps_epi32(ints));
    return _mm_sub_ps(x, _mm_mul_ps(integerpart, mod));
}


void texture_context_to_rgba(const Texture *ctx, uint8_t *dst)
{
    int i;
    int size = ctx->width * ctx->height;

    __m128 t255 = _mm_set1_ps(255.0f);
    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_set1_ps(0);

    for (i = 0; i < size; i+=4)
    {
        __m128i r = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->r[i]), one), zero), t255));
        __m128i g = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->g[i]), one), zero), t255));
        __m128i b = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->b[i]), one), zero), t255));
        __m128i a = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->a[i]), one), zero), t255));

//      r = _mm_slli_epi32(r, 0);
        g = _mm_slli_epi32(g, 8);
        b = _mm_slli_epi32(b, 16);
        a = _mm_slli_epi32(a, 24);

        __m128i result = _mm_or_si128(_mm_or_si128(_mm_or_si128(r,g), b), a);
        _mm_storeu_si128((__m128i*)dst, result);
        dst += 16;
    }
}

static inline __m128i channel_head(__m128i v)
{

    __m128i r = _mm_slli_si128(v, 4);
    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(0,2,0,1));
    return r;
    //return _mm_shuffle_epi32(_mm_slli_si128(v, 6),  _MM_SHUFFLE(2,0,1,0));
    //return _mm_shuffle_epi32(_mm_slli_si128(v, 4), _MM_SHUFFLE(2,0,1,0));;
}

static inline __m128i channel_tail(__m128i v)
{
    __m128i r = _mm_srli_si128(v, 8);
    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(3,1,3,0));
    return r;
    //return _mm_shuffle_epi32(_mm_srli_si128(v, 6), _MM_SHUFFLE(2,3,1,3));;
}

void texture_context_to_rgba16(const Texture *ctx, uint8_t *dst)
{
    int size = ctx->width * ctx->height;
    __m128 t65535 = _mm_set1_ps(65535.0f);
    __m128 one = _mm_set1_ps(1);
    __m128 zero = _mm_set1_ps(0);
    int i;
    for (i = 0; i < size; i+=4)
    {
        __m128i r = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->r[i]), one), zero), t65535));
        __m128i g = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->g[i]), one), zero), t65535));
        __m128i b = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->b[i]), one), zero), t65535));
        __m128i a = _mm_cvtps_epi32(_mm_mul_ps(_mm_max_ps(_mm_min_ps(_mm_load_ps(&ctx->a[i]), one), zero), t65535));

        __m128i r1 = channel_head(r);
        __m128i r2 = channel_tail(r);
        __m128i g1 = channel_head(g);
        __m128i g2 = channel_tail(g);
        __m128i b1 = channel_head(b);
        __m128i b2 = channel_tail(b);
        __m128i a1 = channel_head(a);
        __m128i a2 = channel_tail(a);

        g1 = _mm_slli_si128(g1, 2);
        g2 = _mm_slli_si128(g2, 2);

        b1 = _mm_slli_si128(b1, 4);
        b2 = _mm_slli_si128(b2, 4);

        a1 = _mm_slli_si128(a1, 6);
        a2 = _mm_slli_si128(a2, 6);

        // uint16_t *val = (uint16_t*)&g2;
        // if (_mm_movemask_epi8(_mm_cmpeq_ps(g2, zero)) != 0xFFFF)
        //     printf("value: %d,%d,%d,%d,%d,%d,%d,%d\n", val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);

        __m128i pix = _mm_or_si128(_mm_or_si128(_mm_or_si128(r1,g1), b1), a1);
        _mm_storeu_si128((__m128i*)dst, pix);
        dst += 16;
        pix = _mm_or_si128(_mm_or_si128(_mm_or_si128(r2, g2), b2), a2);
        _mm_storeu_si128((__m128i*)dst, pix);
        dst += 16;

    }
}

void shuffle_test()
{
	__m128i v1 = _mm_set_epi32(3,2,1,0);

    int32_t *val = (int32_t*) &v1;

    printf("\n-----------\n");
    printf("before shuffle: %d,%d,%d,%d\n", val[0], val[1], val[2], val[3]);

    //v1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3,2,1,0));
    val = (int32_t*) &v1;
	printf("after  shuffle: %d,%d,%d,%d\n", val[0], val[1], val[2], val[3]);

}

void shuffle_test2()
{
	//__m128i v1 = _mm_set_epi32(4,3,2,1);
    __m128 t65535 =  _mm_set1_ps(65535.0f);
    //__m128i v = _mm_cvtps_epi32(_mm_mul_ps(_mm_set_ps(1.0f, 1.0f, .5f, .25f), t65535));

    __m128i v = _mm_cvtps_epi32(_mm_mul_ps(_mm_set_ps(1.0f, .9f, .25f, .25f), t65535));
    //__m128i v = _mm_setr_epi16(0, 1, 0, 2, 0, 3, 0, 4);
    uint16_t *val = (uint16_t*) &v;

    printf("\n-----------\n");
    printf("before shuffle: %d,%d,%d,%d,%d,%d,%d,%d\n", val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);

    __m128i v1 = _mm_slli_si128(v, 4);
    v1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(0,2,0,1));
    //v1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3,2,1,0));

    //v1 = _mm_srli_si128(v1, 6);
    val = (uint16_t*) &v1;
	//printf("after  shuffle1: %d,%d,%d,%d,%d,%d,%d,%d\n", val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);


    __m128i v2 = _mm_srli_si128(v, 8);
    v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(3,1,3,0));
    //v1 = _mm_shuffle_epi32(v1, _MM_SHUFFLE(3,2,1,0));

    //v2 = _mm_srli_si128(v2, 4);
    val = (uint16_t*) &v2;
	printf("after  shuffle2: %d,%d,%d,%d,%d,%d,%d,%d\n", val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);


}


int main (int argc, char *argv[])
{
    //printf("hello\n");
    shuffle_test2();
    return 0;
}
