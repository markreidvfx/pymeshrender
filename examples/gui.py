from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
import sys
import meshrender
import meshsmooth
import numpy as np
import numpy
from numpy.linalg import inv,norm
import time
import math
import abcmesh
import os
import av

import cythonmagick

from transformations import euler_matrix, rotation_matrix, decompose_matrix, unit_vector

from concurrent.futures import ThreadPoolExecutor,as_completed

def decode(container):
    for packet in container.demux():
        if packet.stream.type != 'video':
            continue
        for frame in packet.decode():
            yield frame

def read_image_data(path, width, height):
    container = av.open(path)
    for i, frame in enumerate(decode(container)):
        print frame
        rgba = frame.reformat(width, height, "rgba")
        print rgba
        return meshrender.MeshTexture(rgba.planes[0], width, height)

def read_image_data16(path, width, height):
    container = av.open(path)
    for i, frame in enumerate(decode(container)):
        rgba = frame.reformat(width, height, "rgba64le")
        print rgba
        return meshrender.MeshTexture(rgba.planes[0], width, height, depth=16)

def save_image(texture, path):
    i = cythonmagick.Image()
    #print texture.width, texture.height
    #print texture.rgba
    i.from_rawbuffer(bytearray(texture.rgba), texture.width, texture.height, 'BGRA', 'char')
    i.write(path)

def save_image16(texture, path, resize=None):
    i = cythonmagick.Image()
    #print texture.width, texture.height
    #print texture.rgba
    start = time.time()
    i.from_rawbuffer(bytearray(texture.rgba16), texture.width, texture.height, 'BGRA', 'short')
    print "imagemagick read data in %f secs" % (time.time()- start)
    start = time.time()
    print "imagemagick saving..."
    name, ext = os.path.splitext(path)
    if ext.lower() == ".tif":
        i.compress = "lzw"
    elif ext.lower() == '.exr':
        i.compress = "zips"

    if resize:
        i.resize(resize)

    i.write(path)
    print "imagemagick save in %f secs" % (time.time()- start)
def dot(a, b):
    c = np.dot(a,b)

    return [c[0,0], c[0,1], c[0,2]]

def length(v):
    return math.sqrt(v[0] * v[0] +
                     v[1] * v[1] +
                     v[2] * v[2])

def frustum(left, right, bottom, top, znear, zfar):
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)
    # this is transposed incorrectly
    M = np.zeros((4, 4), dtype = np.float32)
    M = np.matrix(M)
    M[0, 0] = +2.0 * znear / (right - left)
    M[0, 2] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[1, 3] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0

    return M

def perspective(fovy, aspect, znear, zfar):
    assert(znear != zfar)
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def rotate_vector(rx, ry, v):
    XM = np.matrix(rotation_matrix(math.radians(rx), [1.0, 0.0, 0.0]))
    YM = np.matrix(rotation_matrix(math.radians(ry), [0.0, 1.0, 0.0]))
    M =   YM * XM

    return  dot(M, v)

def autoclipping(scene_bbox, camera_transform):
    mat4 = np.matrix(camera_transform.T)
    scene_bbox = scene_bbox.dot(mat4)
    scene_bounds =  np.array(abcmesh.bbox_to_bounds(scene_bbox))
    znear = -1 * scene_bounds[1][2]
    zfar =  -1 * scene_bounds[0][2]

    znear = max(znear, 0.01)
    zfar = max(zfar, 1)
    return znear, zfar

class Camera(object):

    def __init__(self):
        self.fovy = 45.0
        self.rotation = (0.0,0.0,0.0)
        self.scale = (1.0,1.0,1.0)
        self.translation = (0.0,0.0, 10.0)
        self.point_of_interest = 90.0
        self.aspect = 1.0
        self.znear = .1
        self.zfar = 1000

        self.width = 1280.0
        self.height = 720.0

    def rotate_matix(self):
        return np.matrix(euler_matrix(math.radians(self.rotation[0]),
                                      math.radians(self.rotation[1]),
                                      math.radians(self.rotation[2]), 'sxyz'), dtype=np.float32)

    def transform_matrix(self):
        T = np.identity(4)
        T[:3, 3] = self.translation
        return np.matrix(T, dtype=np.float32)

    def world_matrix(self):
        return np.dot(self.transform_matrix() , self.rotate_matix())

    def projection_matrix(self):
        return perspective(self.fovy, self.aspect, self.znear, self.zfar)

    def dolly(self, pos):

        v = dot(self.rotate_matix(),[0, 0, -self.point_of_interest, 0])
        view = np.add(self.translation, [v[0],v[1],v[2]])
        v = unit_vector(view)

        t = pos[0] / float(self.width)

        dolly_speed = 5.0

        dolly_by =  1.0 - math.exp(-dolly_speed * t)

        if dolly_by > 1.0:
            dolly_by = 1.0
        elif dolly_by < -1.0:
            dolly_by - 1.0

        dolly_by *= self.point_of_interest
        new_eye = np.add(self.translation, np.multiply(v, dolly_by))
        self.translation = new_eye

        v = np.subtract(new_eye, view)
        self.point_of_interest = length(v)


    def pan(self, pos):
        rotate_matix = self.rotate_matix()

        x = dot(rotate_matix, [1.0, 0.0, 0.0, 0.0])
        y = dot(rotate_matix, [0.0, 1.0, 0.0, 0.0])

        mult_s = 2.0 * self.point_of_interest * math.tan( math.radians( self.fovy ) / 2.0 )
        mult_t = mult_s / float(self.height)
        mult_s /= float(self.width)

        s = -mult_s * pos[0] * 20
        t =  mult_t  * pos[1] * 20

        x = np.multiply(x, s)
        y = np.multiply(y, t)

        self.translation = np.add(self.translation, (x[0], x[1], x[2]))
        self.translation = np.add(self.translation, (y[0], y[1], y[2]))

    def rotate(self, pos):
        rot_x = self.rotation[0]
        rot_y = self.rotation[1]
        rot_z = self.rotation[2]

        v = rotate_vector(rot_x, rot_y, [0.0, 0.0, -self.point_of_interest, 0.0])

        view = np.add([self.translation[0],
                       self.translation[1],
                       self.translation[2]],
                      [v[0],v[1],v[2]])
        speed = 200.0

        rot_y += -speed * (float(pos[0]) / float(self.width))
        rot_x += -speed * (float(pos[1]) / float(self.height))

        v = rotate_vector(rot_x, rot_y, [0.0, 0.0, self.point_of_interest, 0.0])
        new_eye = np.add(view, [v[0],v[1],v[2]])
        self.translation = [new_eye[0], new_eye[1], new_eye[2]]
        self.rotation = [rot_x, rot_y, rot_z]

    def from_matrix(self, matrix):
        scale, shear, angles, translate, perspective = decompose_matrix(matrix)
        self.translation = translate
        self.rotation = [math.degrees(a) for a in angles]


class AbcReader(object):
    def __init__(self, path):
        self.archive = abcmesh.open(path)
        self.camera = None


class Renderer(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object)
    update_frame_range = QtCore.pyqtSignal(object)
    camera_updated = QtCore.pyqtSignal()

    def __init__(self, parent = None, abc=None, texture=None, imageplane=None):
        super(Renderer, self).__init__(parent)
        width = 1920/2
        height = 1080/2
        width = 1280
        height = 720

        self.imageplane = imageplane
        self.save_image_future = None

        #width =1280
        #height = 720

        proj_scale = 8

        self.renderer = meshrender.MeshRenderer(width, height)
        self.projection_renderer = meshrender.MeshRenderer(1024*proj_scale, 1024*proj_scale)
        self.projection_renderer.bake_projection = True
        self.requested_frame = None
        self.requested_checker_size = None
        self.abc = AbcReader(abc)
        self.meshs = {}
        self.smooth_mesh = {}
        self.refiners = {}
        self.mesh_level = {}
        self.prev_level = 0
        self.quit = False
        self.bake_projection = False
        self.render_threads = ThreadPoolExecutor(max_workers=8)
        self.mesh_threads = ThreadPoolExecutor(max_workers=4)
        self.hide = False

        self.camera = Camera()

        self.use_abc_cam = False
        self.camera.aspect = width/ float(height)
        scale = 8
        if texture:
            self.texture = read_image_data(texture, 1024 * scale, 1024 * scale)


    @QtCore.pyqtSlot(object, object)
    def request_frame(self, frame, options):
        checker_size, level, perspective_correction, uvspace, bake_projection, wireframe, bbox, hide = options

        if frame != self.requested_frame or checker_size != self.requested_checker_size or  self.quit:
            return
        self.renderer.checker_size = checker_size
        self.renderer.bbox = bbox
        self.renderer.perspective_correct = perspective_correction
        self.projection_renderer.perspective_correct = perspective_correction
        self.renderer.uvspace = uvspace
        self.bake_projection = bake_projection
        self.hide = hide

        self.renderer.wireframe = wireframe
        self.projection_renderer.wireframe = wireframe
        # self.bake_projection = True

        start = time.time()
        self.renderer.clear()
        if bake_projection:
            self.projection_renderer.clear()
        #print "clear in %.05f secs" % ( time.time() - start)
        #self.draw_triangle(frame)
        start = time.time()
        if not self.hide:
            self.draw_abc(frame, level)
        #print "abc render in %.03f secs" % ( time.time() - start)
        #if frame != self.requested_frame:
        #    return
        self.send_result(frame)

    def draw_mesh(self, renderer, mesh, rect, texture):
        # for item in np.array(mesh.coarse_levels):
        #     print item
        renderer.render_mesh(mesh, rect, texture)

        #print mesh.coarse_edges
        #renderer.render_coarse_edges(mesh, rect, texture)

    def draw_abc(self, frame, level):

        width = self.renderer.width
        height = self.renderer.height

        if self.bake_projection:
            self.projection_renderer.clear()
            width = 1920
            height = 1080

        seconds = frame / 24.0

        render_start = time.time()
        F_MAX = float("+inf")
        F_MIN = float("-inf")
        scene_bounds = np.array( ((F_MAX,F_MAX,F_MAX),
                                  (F_MIN,F_MIN,F_MIN)), dtype=np.float32)

        for mesh in self.abc.archive.mesh_list:
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

        if self.abc.archive.camera_list and self.use_abc_cam:
            camera = self.abc.archive.camera_list[0]
            camera.time = seconds
            camera_transform = inv(np.matrix(camera.world_matrix, dtype=np.float32))

            znear,zfar = autoclipping(scene_bbox, camera_transform)
            print "zcliping", znear, zfar

            persp_matrix = np.matrix(camera.perspective_matrix(width/float(height), znear=znear, zfar=zfar), dtype=np.float32)
            self.camera.from_matrix(inv(camera_transform))
            self.camera.fovy = camera.fovy
            self.camera.znear = znear
            self.camera.zfar = zfar
            self.camera_updated.emit()
        else:
            camera_transform = inv(self.camera.world_matrix())

            znear,zfar = autoclipping(scene_bbox, camera_transform)
            print "zcliping", znear, zfar

            self.camera.znear = znear
            self.camera.zfar = zfar
            self.camera_updated.emit()
            persp_matrix = self.camera.projection_matrix()


        #viewport_matrix = np.matrix(abcmesh.viewport_matrix(width, height))
        viewport_matrix = np.matrix(meshrender.viewport_matrix(width-1, height-1), dtype=np.float32)
        #level = 0
        mesh_futures = []

        for mesh in self.abc.archive.mesh_list[:1]:
            f = self.mesh_threads.submit(self.process_mesh, mesh, persp_matrix, camera_transform, viewport_matrix, seconds, level)
            mesh_futures.append(f)

        renderer = self.renderer
        texture = self.texture

        x_tiles = 4
        y_tiles = 2
        if self.bake_projection and self.imageplane:
            texture = read_image_data16( self.imageplane % frame, width, height)
            renderer = self.projection_renderer
            x_tiles = 2
            y_tiles = 2

        for f in as_completed(mesh_futures):
            mesh_name, mesh = f.result()

            x_tile_size = renderer.width / x_tiles
            y_tile_size = renderer.height / y_tiles

            minx = 0
            maxx = x_tile_size

            miny = 0
            maxy = y_tile_size
            render_futures = []

            #
            # if self.bake_projection:
            #     renderer.render_mesh(mesh, texture=texture)
            #     continue

            for y in range(y_tiles):
                minx = 0
                maxx = x_tile_size

                for x in range(x_tiles):
                    rect = [(minx, miny), (maxx-1, maxy-1)]
                    #print rect
                    r_future = self.render_threads.submit(self.draw_mesh, renderer, mesh, rect, texture)
                    render_futures.append(r_future)

                    minx += x_tile_size
                    maxx += x_tile_size

                miny += y_tile_size
                maxy += y_tile_size

            for r_future in as_completed(render_futures):
                r_future.result()

        if self.bake_projection:
            print "texture bake in %f secs" % (time.time()- render_start)

            grow_start = time.time()
            self.projection_renderer.grow()
            print "texture grow in %f secs" % (time.time()- grow_start)
            # self.texture = self.projection_renderer.to_texture()

            self.texture = self.projection_renderer.to_texture( self.projection_renderer.width/2,  self.projection_renderer.height/2)


            if self.save_image_future:
                self.save_image_future.result()
                self.save_image_future = None

            #self.save_image_future = self.mesh_threads.submit( save_image16, self.texture, "out.tif", "4096x4096")
            #save_image16(self.texture, "out.tif", "1024x1024")
            # save_image16(self.texture, "out.tif", "4096x4096")


            self.bake_projection = False
            self.draw_abc(frame, level)
        else:
            print "render in %f secs" % (time.time() - render_start)

    def send_result(self, frame):
        start = time.time()
        width = self.renderer.width
        height = self.renderer.height

        if not self.renderer.uvspace:
            bg_image = read_image_data( self.imageplane % frame, width, height)
            self.renderer.under(bg_image)

        image_data = self.renderer.rgba
        #print "image converted in %.04f secs" % ( time.time() - start)

        bytesPerPixel = 4
        #Format_RGB32
        #Format_ARGB32_Premultiplied
        image = QtGui.QImage(image_data, width,  height, width * bytesPerPixel,
                             QtGui.QImage.Format_RGB32)


        if self.quit:
            return

        self.frame_ready.emit(image)
        #print "image sent in %.03f secs" % ( time.time() - start)

    def process_mesh(self, mesh, persp_matrix, camera_transform, viewport_matrix, seconds, level):
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

        elif mesh_name in self.smooth_mesh and self.mesh_level[mesh_name] == level:
            smooth_mesh = self.smooth_mesh[mesh_name]
            refiner = self.refiners[mesh_name]

            refiner.mesh.vertices.values =  mesh.vertices_vec4
            normal_values = mesh.normal_values
            if normal_values:
                refiner.mesh.fvchannels[1].values = normal_values

            smooth_mesh = refiner.refine_uniform(level, smooth_mesh, generate_indices=False)
            self.smooth_mesh[mesh_name] = smooth_mesh
            rmesh = meshrender.Mesh()
            normals = None
            if normal_values:
                normals = smooth_mesh.fvchannels[1]
            rmesh.from_meshsmooth(smooth_mesh, smooth_mesh.fvchannels[0], normals)

            #rmesh.coarse_edges = refiner.refine_coarse_edges(level)
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

            # "NONE"
            # "EDGE_ONLY"
            # "EDGE_AND_CORNER"
            #
            # "LINEAR_NONE"
            # "LINEAR_CORNERS_ONLY"
            # "LINEAR_CORNERS_PLUS1"
            # "LINEAR_CORNERS_PLUS2"
            # "LINEAR_BOUNDARIES"
            # "LINEAR_ALL"
            boundry = "EDGE_ONLY"
            boundry = "EDGE_AND_CORNER"
            # boundry = "NONE"
            fvar_interp = "LINEAR_BOUNDARIES"
            #fvar_interp = "LINEAR_CORNERS_ONLY"
            refiner = meshsmooth.TopologyRefiner(src_mesh, BoundaryInterpolation = boundry,
                                                           FVarInterpolation = fvar_interp)
            smooth_mesh = refiner.refine_uniform(level)

            self.smooth_mesh[mesh_name] = smooth_mesh
            self.refiners[mesh_name] = refiner
            self.mesh_level[mesh_name] = level

            rmesh = meshrender.Mesh()
            normals = None
            if normal_indices:
                normals = smooth_mesh.fvchannels[1]
            rmesh.from_meshsmooth(smooth_mesh, smooth_mesh.fvchannels[0], normals)
            #rmesh.coarse_edges = refiner.refine_coarse_edges(level)

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

        return mesh_name, rmesh

    def draw_triangle(self, angle):
        width = self.renderer.width
        height = self.renderer.height

        face = [3]
        indices = range(3)
        verts = [(0.5, 0.0, 0.0, 0.0),
                 (1.0, 1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0, 0.0)]
        normals = [(1.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0)]

        uvs = [(1.0, 0.5),
               (0.0, 1.0),
               (0.0, 0.0)]

        mesh = meshrender.Mesh()
        mesh.face_counts = np.array(face,  dtype=np.int32)
        mesh.face_indices = np.array(indices,  dtype=np.int32)
        mesh.uv_indices = mesh.face_indices
        mesh.normal_indices = mesh.face_indices

        verts =  np.array(verts, dtype=np.float32)

        verts[:] = np.multiply(verts, (width, height, 1.0, 1.0))

        mesh.vertices = np.array(verts, dtype=np.float32)
        mesh.uvs = np.array(uvs, dtype=np.float32)
        mesh.normals = np.array(normals, dtype=np.float32)

        self.renderer.render_mesh(mesh)

class RenderWidget(QtGui.QWidget):

    request_frame = QtCore.pyqtSignal(object, object)
    def __init__(self, parent=None, abc=None, texture=None, imageplane=None):
        super(RenderWidget, self).__init__(parent)
        self.display = QtGui.QLabel()
        self.timeline = QtGui.QScrollBar(Qt.Horizontal)
        self.frame_control = QtGui.QSpinBox()
        self.checker_control = QtGui.QSpinBox()
        self.checker_control.setValue(15)
        self.subdiv_control = QtGui.QSpinBox()
        self.subdiv_control.setValue(1)

        self.perspective_correct = QtGui.QCheckBox("persp")
        self.perspective_correct.setChecked(True)

        self.uvspace = QtGui.QCheckBox("uvspace")
        self.uvspace.setChecked(False)

        self.wireframe = QtGui.QCheckBox("wireframe")
        self.wireframe.setChecked(False)

        self.bbox = QtGui.QCheckBox("bbox")
        self.bbox.setChecked(False)

        self.hide = QtGui.QCheckBox("hide")
        self.hide.setChecked(False)

        camera_controls_layout = QtGui.QVBoxLayout()

        self.use_abc_cam = QtGui.QCheckBox("use abc cam")
        self.use_abc_cam.setChecked(True)

        camera_controls_layout.addWidget(self.use_abc_cam)

        self.bake_projection_button = QtGui.QPushButton("Bake Projection")

        self.camera_controls =  []

        self.cam = {}
        for group, names in ( ("translate", ('x', 'y', 'z')),
                              ('rotate', ('x', 'y', 'z')),
                              ("other", ('fovy', 'near', 'far', 'point_of_interest')) ):
            l = QtGui.QHBoxLayout()

            l.addWidget(QtGui.QLabel(group))
            self.cam[group] = []
            for n in names:
                c = QtGui.QDoubleSpinBox()
                c.setRange(-999999999, 999999999)
                c.setDecimals(4)
                #c.setSingleStep(.1)
                l.addWidget(QtGui.QLabel(n))
                l.addWidget(c)
                c.valueChanged.connect(lambda x: self.update_camera())
                self.camera_controls.append(c)
                self.cam[group].append(c)
            l.addStretch()
            camera_controls_layout.addLayout(l)


        layout = QtGui.QVBoxLayout()

        toplayout  = QtGui.QHBoxLayout()
        toplayout.addWidget(self.display)
        #toplayout.addLayout(camera_controls_layout)

        layout.addLayout(toplayout)

        control_layout =  QtGui.QHBoxLayout()
        control_layout.addWidget(self.timeline)
        control_layout.addWidget(self.frame_control)
        control_layout.addWidget(self.checker_control)
        control_layout.addWidget(self.subdiv_control)
        control_layout.addWidget(self.perspective_correct)
        control_layout.addWidget(self.uvspace)
        control_layout.addWidget(self.wireframe)
        control_layout.addWidget(self.bbox)
        control_layout.addWidget(self.hide)
        control_layout.addWidget(self.bake_projection_button)
        layout.addLayout(control_layout)
        layout.addLayout(camera_controls_layout)
        self.setLayout(layout)

        self.timeline.valueChanged.connect(self.frame_changed)
        self.checker_control.valueChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.subdiv_control.valueChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.perspective_correct.stateChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.uvspace.stateChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.wireframe.stateChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.bbox.stateChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))
        self.hide.stateChanged.connect(lambda x: self.frame_changed(self.frame_control.value()))

        self.bake_projection_button.clicked.connect(lambda x: self.frame_changed(self.frame_control.value(), True))

        self.use_abc_cam.stateChanged.connect(lambda x: self.update_camera())

        self.frame_control.valueChanged.connect(self.frame_changed)


        self.renderer = Renderer(abc=abc, texture=texture, imageplane=imageplane)

        start = self.renderer.abc.archive.start_time *24
        end = self.renderer.abc.archive.end_time *24
        self.frame_control.setRange(start,end)
        self.timeline.setRange(start,end)
        self.request_frame.connect(self.renderer.request_frame)
        self.renderer.camera_updated.connect(self.reload_camera)

        use_thread = True
        if use_thread:
            self.renderer_thread = QtCore.QThread()
            self.renderer.moveToThread(self.renderer_thread)
            self.renderer_thread.start()
            self.renderer.frame_ready.connect(self.frame_ready)
        else:
            self.renderer.frame_ready.connect(self.frame_ready)

        self.timer = QtCore.QTimer()
        def crazy_loop():
            v = self.frame_control.value() + 1
            if v > self.frame_control.maximum():
                v = self.frame_control.minimum()
            self.frame_control.setValue(v)

        self.timer.timeout.connect(crazy_loop)
        # self.timer.start(1000/ 24);
        self.prev_time = time.time()
        self.frame_count = 0
        self.frame_changed(0)

        t = self.cam['translate']
        r = self.cam['rotate']
        o = self.cam['other']

        for c,v in zip( t, (-46.372, 58.606, 286.432)):
            c.setValue(v)
        for c,v in zip( r, (-4.33, -10.441, -3.035)):
            c.setValue(v)

        for c,v in zip( o, (4.34826587841, .1, 1000)):
            c.setValue(v)

        o[3].setValue(90)

        self.mouse_buttons = Qt.NoButton
        self.mouse_time = None
        self.mouse_pos = None

    def mousePressEvent(self, event):

        self.mouse_buttons = event.buttons()
        self.mouse_time = time.time()
        self.mouse_pos = self.display.mapFromGlobal(event.globalPos())

    def mouseMoveEvent(self, event):

        if self.mouse_buttons == Qt.NoButton:
            event.ignore()
            return

        time_delta = time.time()- self.mouse_time

        if time_delta < .05:
            event.ignore()
            return

        cur = self.display.mapFromGlobal(event.globalPos())

        last = self.mouse_pos

        self.mouse_time = time.time()
        self.mouse_pos = cur

        x_delta = (cur.x() - last.x() ) * time_delta
        y_delta = (cur.y() - last.y() ) * time_delta

        if (self.mouse_buttons & Qt.RightButton and event.modifiers() & Qt.AltModifier) or \
                            (self.mouse_buttons & Qt.LeftButton and self.mouse_buttons & Qt.MiddleButton and event.modifiers() & Qt.AltModifier):
            print "dolly"
            self.renderer.camera.dolly([x_delta, y_delta])
            self.reload_camera()
            self.update_camera()

        elif (self.mouse_buttons & Qt.LeftButton and event.modifiers() & Qt.AltModifier and event.modifiers() & Qt.ControlModifier) or \
                        (self.mouse_buttons & Qt.MiddleButton and event.modifiers() & Qt.AltModifier):
            print "pan"
            self.renderer.camera.pan([x_delta, y_delta])
            self.reload_camera()
            self.update_camera()

        elif self.mouse_buttons & Qt.LeftButton and event.modifiers() & Qt.AltModifier:
            print "rotate"
            self.renderer.camera.rotate([x_delta, y_delta])
            self.reload_camera()
            self.update_camera()

        event.accept()


    def mouseReleaseEvent(self, event):

        self.mouse_buttons = Qt.NoButton
        #for c
        #self.cam['fov'].setValue(4.34826587841)

    def frame_changed(self, value, bake=False):
        self.timeline.blockSignals(True)
        self.frame_control.blockSignals(True)

        self.timeline.setValue(value)
        self.frame_control.setValue(value)

        self.timeline.blockSignals(False)
        self.frame_control.blockSignals(False)
        self.renderer.requested_frame = value

        persp = self.perspective_correct.isChecked()
        uvspace = self.uvspace.isChecked()
        checker_size = self.checker_control.value()
        wireframe = self.wireframe.isChecked()
        bbox = self.bbox.isChecked()
        hide = self.hide.isChecked()

        level = self.subdiv_control.value()

        self.renderer.requested_checker_size = checker_size
        self.request_frame.emit(value, (checker_size, level, persp, uvspace, bake, wireframe, bbox, hide))

    @QtCore.pyqtSlot(object)
    def frame_ready(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.display.setPixmap(pixmap)

        if not (self.frame_count % 10):
            t = time.time()
            print "%0.02f fps" % (self.frame_count / ( t - self.prev_time))
            self.frame_count = 0
            self.prev_time = t

        self.frame_count+= 1

    def closeEvent(self, event):
        self.renderer.quit = True

        self.renderer_thread.quit()
        self.renderer_thread.wait()

        event.accept()

    @QtCore.pyqtSlot()
    def reload_camera(self):
        # print "camera reload"
        for c in self.camera_controls:
            c.blockSignals(True)
        camera = self.renderer.camera

        for c, v in zip(self.cam['translate'], camera.translation):
            c.setValue(v)

        for c, v in zip(self.cam['rotate'], camera.rotation):
            c.setValue(v)


        self.cam['other'][0].setValue(camera.fovy)
        self.cam['other'][1].setValue(camera.znear)
        self.cam['other'][2].setValue(camera.zfar)
        self.cam['other'][3].setValue(camera.point_of_interest)

        for c in self.camera_controls:
            c.blockSignals(False)


    def update_camera(self):
        # print "camera_update"
        self.renderer.use_abc_cam = self.use_abc_cam.isChecked()
        for c in self.camera_controls:
            c.blockSignals(True)
        translate = [c.value() for c in self.cam['translate']]
        self.renderer.camera.translation = translate

        rotate = [c.value() for c in self.cam['rotate']]
        self.renderer.camera.rotation = rotate

        other = [c.value() for c in self.cam['other']]


        self.renderer.camera.fovy = other[0]
        self.renderer.camera.znear = other[1]
        self.renderer.camera.zfar = other[2]
        self.renderer.camera.point_of_interest = other[3]

        for c in self.camera_controls:
            c.blockSignals(False)

        self.frame_changed(self.frame_control.value())

if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--abc", dest="abc", metavar="FILE")
    parser.add_option("--texture", dest="texture", metavar="FILE")
    parser.add_option("--imageplane", dest="imageplane", metavar="FILE")

    (options, args) = parser.parse_args()

    app = QtGui.QApplication(sys.argv)
    window = RenderWidget(abc=options.abc,
                          texture=options.texture,
                          imageplane=options.imageplane)
    #test_file = sys.argv[1]
    #window.set_file(test_file)
    window.show()
    window.raise_()
    sys.exit(app.exec_())
