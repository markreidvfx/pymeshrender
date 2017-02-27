import time
import os

import numpy as np
from numpy.linalg import inv, norm

import abcmesh
import meshrender
import meshsmooth
import cythonmagick
import av

F_MAX = float("+inf")
F_MIN = float("-inf")
from concurrent.futures import ThreadPoolExecutor,as_completed


def read_image_data16(path, width=None, height=None):
    container = av.open(path)
    for i, frame in enumerate(container.decode(video=0)):
        if not width:
            width = frame.width
        if not height:
            height = frame.height

        rgba = frame.reformat(width, height, format="rgba64le")
        return meshrender.MeshTexture(rgba.planes[0], width, height, depth=16)

def save_exr(texture, path):
    import OpenEXR
    import Imath
    start = time.time()
    hdr = OpenEXR.Header(texture.width, texture.height)

    pixels = {}
    channels = {}
    chan_format = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    # chan_format = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    c_start = time.time()
    for c, data in [("R", texture.r_f16), ("G",texture.g_f16), ("B", texture.b_f16), ("A", texture.a_f16)]:
        # pixels[c] = np.array(data).astype(np.float16)
        pixels[c] = data
        channels[c]= chan_format
    print "channel convert in %f secs" % (time.time() - c_start)

    hdr['channels'] = channels
    hdr['compression'] = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
    exr = OpenEXR.OutputFile(path, hdr)
    exr.writePixels(pixels)
    exr.close()

    print "wrote exr in %f secs" % ( time.time() - start)

def save_image16(texture, path, resize=None):


    with ThreadPoolExecutor(max_workers=16) as e:
        for i in xrange(1):
            start = time.time()
            # texture = texture.grow()

            texture = mulithread_texture_grow(e, texture)
            print "texture grow in %f secs" % (time.time() - start)


    start = time.time()
    texture = texture.resample(texture.width/2, texture.height/2)
    print "resize in %f secs" % (time.time() - start)

    print texture.width, texture.height

    save_exr(texture, path)
    return
    #print texture.rgba
    start = time.time()
    i = cythonmagick.Image()
    i.verbose=True
    i.from_rawbuffer(bytearray(texture.rgba16), texture.width, texture.height, 'RGBA', 'short')
    print "imagemagick read data in %f secs" % (time.time()- start)
    # start = time.time()

    i.attributes['colorspace'] = "RGB"

    name, ext = os.path.splitext(path)
    if resize:
        i.filter = "Catrom"
        # start = time.time()

        #i.distort("resize", [4096,4096])
        # print i.size()
        # i.scale(resize)
        #i.resize(resize)
        # print "resize in %f secs" % (time.time() - start)

    if ext.lower() == ".tif":
        i.compress = "lzw"


    elif ext.lower() == '.exr':
        # start = time.time()
        # alpha = i.channel("alpha")
        # alpha.negate()
        # i.type = "TrueColor"
        # i.composite(alpha, "multiply")
        # i.composite(alpha, "copyopacity")
        # print "premult in %f secs" % (time.time() - start)
        i.compress = "zips"


    start = time.time()
    i.write(path)
    print "imagemagick save in %f secs" % (time.time()- start)

def autoclipping(scene_bbox, camera_transform):
    mat4 = np.matrix(camera_transform.T)
    scene_bbox = scene_bbox.dot(mat4)
    scene_bounds =  np.array(abcmesh.bbox_to_bounds(scene_bbox))
    znear = -1 * scene_bounds[1][2]
    zfar =  -1 * scene_bounds[0][2]

    znear = max(znear, 0.01)
    zfar = max(zfar, 1)
    return znear, zfar

def process_mesh(mesh, persp_matrix, camera_transform, viewport_matrix, seconds, level):
    start  = time.time()
    mesh_name = mesh.full_name
    mesh.time = seconds
    bbox = np.array(mesh.bbox)

    if level == 0:
        rmesh = meshrender.Mesh()
        rmesh.face_counts = mesh.face_counts
        rmesh.face_indices = mesh.face_indices
        rmesh.uvs = mesh.uv_values
        rmesh.uv_indices = np.array(mesh.uv_indices, dtype=np.int32)

        normal_indices = mesh.normal_indices
        rmesh.normal_indices = None
        if normal_indices:
            rmesh.normal_indices = np.array(mesh.normal_indices, dtype=np.int32)
            rmesh.normals = mesh.normal_values
        rmesh.vertices = mesh.vertices_vec4

    else:
        channels = []
        uv_indices = mesh.uv_indices
        if uv_indices:
            channels.append(meshsmooth.FVarChannel("uvs", np.array(uv_indices,  dtype=np.int32),
                                                  mesh.uv_values))

        normal_indices = mesh.normal_indices
        if normal_indices:
            channels.append(meshsmooth.FVarChannel("normals", np.array(mesh.normal_indices, dtype=np.int32),
                                                      mesh.normal_values))

        vert_channel = meshsmooth.FVarChannel("verts", mesh.face_indices, mesh.vertices_vec4)
        src_mesh =  meshsmooth.Mesh(mesh.face_counts, vert_channel, channels)

        # "LINEAR_NONE"
        # "LINEAR_CORNERS_ONLY"
        # "LINEAR_CORNERS_PLUS1"
        # "LINEAR_CORNERS_PLUS2"
        # "LINEAR_BOUNDARIES"
        # "LINEAR_ALL"

        boundry = "NONE"
        # boundry = "EDGE_ONLY"
        boundry = "EDGE_AND_CORNER"


        fvar_interp = "LINEAR_NONE"
        fvar_interp = "LINEAR_CORNERS_ONLY"
        # fvar_interp = "LINEAR_CORNERS_PLUS2"
        fvar_interp = "LINEAR_BOUNDARIES"
        # fvar_interp = "LINEAR_ALL"

        refiner = meshsmooth.TopologyRefiner(src_mesh, BoundaryInterpolation = boundry,
                                                       FVarInterpolation = fvar_interp)
        smooth_mesh = refiner.refine_uniform(level)

        rmesh = meshrender.Mesh()
        normals = None
        if normal_indices:
            normals = smooth_mesh.fvchannels[1]
        rmesh.from_meshsmooth(smooth_mesh, smooth_mesh.fvchannels[0], normals)

    model_transform = np.matrix(mesh.world_matrix, dtype=np.float32)

    mat4 = viewport_matrix * persp_matrix * camera_transform * model_transform
    mat4 = np.matrix(mat4.T,  dtype=np.float32)

    v  = np.array(rmesh.vertices)

    rmesh.objspace_vertices = np.empty(v.shape, dtype=np.float32)
    rmesh.objspace_vertices[:] = v

    rmesh.vertices = v.dot(mat4)
    meshrender.perpsective_divide(rmesh.vertices)

    rmesh.bbox = bbox.dot(mat4)
    meshrender.perpsective_divide(rmesh.bbox)

    return rmesh

def mulithread_render_mesh(executor, renderer, mesh, x_tiles=4, y_tiles=4, texture=None):

    x_tile_size = renderer.width / x_tiles
    y_tile_size = renderer.height / y_tiles

    miny = 0
    maxy = y_tile_size
    render_futures = []

    for y in range(y_tiles):
        minx = 0
        maxx = x_tile_size

        for x in range(x_tiles):
            rect = [(minx, miny), (maxx-1, maxy-1)]
            render_futures.append(executor.submit(renderer.render_mesh, mesh, rect=rect, texture=texture))
            # renderer.render_mesh(mesh, rect=rect, texture=texture)

            minx += x_tile_size
            maxx += x_tile_size

        miny += y_tile_size
        maxy += y_tile_size

    for r_future in as_completed(render_futures):
        r_future.result()

def mulithread_texture_grow(executor, texture, x_tiles=4, y_tiles=4):

    x_tile_size = texture.width / x_tiles
    y_tile_size = texture.height / y_tiles

    miny = 0
    maxy = y_tile_size
    render_futures = []

    dest_texture = meshrender.MeshTexture(None, texture.width, texture.height)

    for y in range(y_tiles):
        minx = 0
        maxx = x_tile_size

        for x in range(x_tiles):
            rect = [(minx, miny), (maxx-1, maxy-1)]



            # print texture.width, texture.height
            # print rect
            # texture.grow(16, rect=rect, dst_texture=dest_texture)
            # if x % 2 == 0:
            render_futures.append(executor.submit(texture.grow, 16, rect=rect, dst_texture=dest_texture))
            # render_futures.append(executor.submit(renderer.render_mesh, mesh, rect=rect, texture=texture))
            # renderer.render_mesh(mesh, rect=rect, texture=texture)

            minx += x_tile_size
            maxx += x_tile_size

            # return dest_texture

        miny += y_tile_size
        maxy += y_tile_size

    for r_future in as_completed(render_futures):
        r_future.result()

    return dest_texture

def project3d(abc_path, image_path, dest_path, frame=None, subdiv_level=0, size=1024):
    start = time.time()
    abc = abcmesh.open(abc_path)
    if not abc.camera_list:
        raise Exception("abc files has no cameras")

    camera = abc.camera_list[0]

    if frame is None:
        frame = 1

    seconds = frame / 24.0

    image_texture = read_image_data16(image_path)

    width = image_texture.width
    height = image_texture.height

    scale = 8
    renderer = meshrender.MeshRenderer(1024 * scale, 1024 * scale)
    renderer.bake_projection = True
    renderer.uvspace = True
    renderer.wireframe = True
    scene_bounds = np.array(((F_MAX,F_MAX,F_MAX),
                             (F_MIN,F_MIN,F_MIN)), dtype=np.float32)
    for mesh in abc.mesh_list:
        mesh.time = seconds

        model_transform = np.matrix(mesh.world_matrix)

        mat4 = np.matrix(model_transform.T)
        bbox = np.array(mesh.bbox)
        bbox = bbox.dot(mat4)

        mesh_bounds = abcmesh.bbox_to_bounds(bbox)
        # m_bounds = mesh.bounds
        for i in range(3):
            scene_bounds[0][i] = min(scene_bounds[0][i], mesh_bounds[0][i])
        for i in range(3):
            scene_bounds[1][i] = max(scene_bounds[1][i], mesh_bounds[1][i])

    scene_bbox = np.array(abcmesh.bounds_to_bbox(scene_bounds))

    camera.time = seconds
    camera_transform = inv(np.matrix(camera.world_matrix, dtype=np.float32))

    znear, zfar = autoclipping(scene_bbox, camera_transform)

    persp_matrix = np.matrix(camera.perspective_matrix(width/float(height),
                             znear=znear, zfar=zfar), dtype=np.float32)

    viewport_matrix = np.matrix(meshrender.viewport_matrix(width-1, height-1), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=16) as e:

        for mesh in abc.mesh_list[:1]:
            rmesh = process_mesh(mesh, persp_matrix,
                                 camera_transform, viewport_matrix, seconds, subdiv_level)

            # renderer.render_mesh(rmesh, texture=image_texture)
            mulithread_render_mesh(e, renderer, rmesh,  texture=image_texture)

    print "projection rendered in %f secs"% ( time.time() - start)

    save_image16(renderer, dest_path, "%dx%d" % (size, size))

if __name__ == "__main__":

    from optparse import OptionParser

    default_size = 1024
    # default_size = 1024
    parser = OptionParser()
    parser.add_option("--abc", dest="abc", metavar="FILE")
    parser.add_option("--image", dest="projected_image", metavar="FILE")
    parser.add_option("--overlay", dest="texture_image", metavar="FILE")
    parser.add_option("--frame", dest="frame", type="int", default=1)
    parser.add_option("--level", dest="level", type="int", help="subdiv level", default=1)
    parser.add_option("--size", dest="size", type="int", help="dest image size", default=default_size)

    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("not enought args")

    start = time.time()
    project3d(options.abc, options.projected_image, args[0],
              frame = options.frame,
              subdiv_level = options.level,
              size = options.size
    )
    print "projection in %f secs" % (time.time() - start)
