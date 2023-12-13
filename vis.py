import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from collections import defaultdict
import numpy as np

class ArtistHandler:
    def init(self, ax, obj):
        raise NotImplementedError()
    def update(self, obj):
        raise NotImplementedError()
    def remove(self):
        raise NotImplementedError()

class BaseArtistHandler(ArtistHandler):
    def __init__(self):
        self.artist = None
    def inactive(self):
        return self.artist is None
    def get_artists(self):
        return self.artist,
    def remove(self):
        if not self.inactive():
            self.artist.remove()
            self.artist = None

class PointsArtistHandler(BaseArtistHandler):
    def __init__(self, plot_kwargs):
        super().__init__()
        self.plot_kwargs = plot_kwargs
        self.is_3d = False

    def init(self, ax, points):
        if points.shape[1] == 3:
            self.artist = ax.scatter(points[:, 0], points[:, 1], points[:, 2], **self.plot_kwargs)
            self.is_3d = True
        else:
            self.artist = ax.plot(points[:, 0], points[:, 1], marker="o", ls="", **self.plot_kwargs)[0]
            self.is_3d = False

    def update(self, points):
        if self.is_3d:
            self.artist._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        else:
            self.artist.set_data(points[:, 0], points[:, 1])
class BaseMultipleArtistHandler(ArtistHandler):
    def __init__(self):
        super().__init__()
        self.artists = []
    def inactive(self):
        return len(self.artists) == 0
    def get_artists(self):
        return self.artists
    def remove(self):
        if not self.inactive():
            for artist in self.artists:
                artist.remove()
            self.artists = []
class CameraConeArtistHandler(BaseMultipleArtistHandler):
    def __init__(self, edge_length=100, focal_length=1):
        super().__init__()
        self.edge_length = edge_length
        self.focal_length = focal_length

    def get_camera_cone_edges(self, world2cam, z):
        pix_list = np.array([
            [-1, 1],
            [1, 1],
            [1, -1],
            [-1, -1],
        ]) / self.focal_length
        if world2cam.shape[0] == 3:
            world2cam = np.concatenate((world2cam, np.array([0, 0, 0, 1])[None]), axis=0)
        pix_list = np.concatenate([pix_list, np.ones((4, 1))], axis=1)
        pix_list_unprojected = pix_list * z
        pix_list_unprojected = np.concatenate([pix_list_unprojected, np.ones((4, 1))], axis=1)
        world_point_list = pix_list_unprojected @ np.linalg.inv(world2cam).T
        camera_pos = -np.linalg.inv(world2cam[:3, :3]) @ world2cam[:3, 3:]
        world_point_list = world_point_list[:, :3]
        edges = []
        for i in range(4):
            edges.append([world_point_list[i], world_point_list[(i + 1) % 4]])
            edges.append([camera_pos[:, 0], world_point_list[i]])
        return edges
    def init(self, ax, RT):
        edges = self.get_camera_cone_edges(RT, self.edge_length)
        for e in edges:
            self.artists.append(
                ax.plot(*np.array(e).T, color='g', ls='-')[0]
            )

    def update(self, RT):
        edges = self.get_camera_cone_edges(RT, self.edge_length)
        for i, e in enumerate(edges):
            self.artists[i].set_data(np.array(e)[:, :2].T)
            self.artists[i].set_3d_properties(np.array(e)[:, 2].T)



class ImageArtistHandler(BaseArtistHandler):
    def init(self, ax, image):
        ax.grid(False)
        self.artist = ax.imshow(image)
    def update(self, image):
        self.artist.set_data(image)

class LabelArtistHandler(BaseArtistHandler):
    def __init__(self, color='red', formatting='{}', ha='center', va='bottom', fontsize=12):
        super().__init__()
        self.text_kwargs = dict(color=color, ha=ha, va=va, fontsize=fontsize)
        self.formatting = formatting
    def init(self, ax, p):
        pos, label = p
        self.artist = ax.text(pos[0], pos[1], label, **self.text_kwargs)
    def update(self, p):
        pos, label = p
        self.artist.set_text(label)
        self.artist.set_position(pos)

class BBoxArtistHandler(BaseArtistHandler):
    def __init__(self, color):
        super().__init__()
        self.color = color
    def init(self, ax, bbox):
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
            linewidth=2, 
            edgecolor=self.color, 
            facecolor='none'
        )
        self.artist = ax.add_patch(rect)
    def update(self, bbox):
        self.artist.set_bounds(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])


class VideoGrid:
    def __init__(self, *, rows: int, cols: int, frame_num: int, fps: float, figsize=None):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=figsize)
        self.axs = [[
            self.fig.add_subplot(rows, cols, row * cols + col + 1) for col in range(cols)
        ] for row in range(rows)]
        self.frame_num = frame_num
        self.fps = fps
        self.updaters = []
        self.init_colormaps()
        self.object_store = defaultdict(list)
        self.is_3d = [[False for _ in range(cols)] for _ in range(rows)]
    def set_3d_projection(self, row: int, col: int, vertical_axis='z', azim=None, elev=None):
        self.axs[row][col].remove()
        self.axs[row][col] = self.fig.add_subplot(self.rows, self.cols, row * self.cols + col + 1, projection='3d')
        self.axs[row][col].view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
        self.axs[row][col].set_xlabel('X')
        self.axs[row][col].set_ylabel('Y')
        self.axs[row][col].set_zlabel('Z')
        self.is_3d[row][col] = True

    def set_lim(self, row: int, col: int, xlim=None, ylim=None, zlim=None):
        ax = self.axs[row][col]
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)
    def set_lim_from_radius(self, row: int, col: int, radius, center=None):
        ax = self.axs[row][col]
        if center is None:
            center = np.array([0, 0, 0])
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        if ax.name == '3d':
            ax.set_zlim(center[2] - radius, center[2] + radius)
    def generate_id(self, obj_type):
        cur_obj_list = self.object_store[obj_type]
        idx = len(cur_obj_list)
        cur_obj_list.append(None)
        return idx

    def init_colormaps(self):
        self.colormaps = defaultdict(lambda: plt.get_cmap('hsv'))

    def get_color(self, obj_type, obj_key):
        # v = obj_key / (len(self.object_store[obj_type]) + 1)
        return self.colormaps[obj_type](np.random.uniform())

        
    def update(self, *args, **kwargs):
        artists = []
        for updater in self.updaters:
            artists += list(updater(*args, **kwargs))

        # Print all lims
        for row in range(self.rows):
            for col in range(self.cols):
                # Set axis ranges to be equal
                ax = self.axs[row][col]
                if not self.is_3d[row][col]:
                    ax.set_aspect('equal')
                else:
                    xlims, ylims, zlims = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
                    xcenter, ycenter, zcenter = (xlims[0] + xlims[1]) / 2, (ylims[0] + ylims[1]) / 2, (zlims[0] + zlims[1]) / 2
                    radius = max(xlims[1] - xlims[0], ylims[1] - ylims[0], zlims[1] - zlims[0]) / 2
                    ax.set_xlim(xcenter - radius, xcenter + radius)
                    ax.set_ylim(ycenter - radius, ycenter + radius)
                    ax.set_zlim(zcenter - radius, zcenter + radius)
        return artists

    def add_video(self, row, col, imgs):
        """
        Input:
            idx: int, the column number
            imgs: (N, H, W[, 3]), a video to add to the column
        """
        self.add_artist_handler(
            row, col,
            'image',
            ImageArtistHandler(),
            imgs
        )

    def title(self, row, col, title):
        self.axs[row][col].set_title(title)
    def add_cameras(self, row, col, RT, focal_length=None, edge_length=100):
        """
        Input:
            idx: int, the column number
            RT: (N, 3, 4), a list of camera extrinsic matrices
        """
        self.add_artist_handler(
            row, col,
            'cameras',
            CameraConeArtistHandler(edge_length=edge_length, focal_length=focal_length),
            RT
        )
    def add_points(self, row, col, points, plot_kwargs={}):
        """
        Input:
            idx: int, the column number
            points: list of (N1, 2), (N2, 2), ... of 2D points to scatter on column 2
        """
        self.add_artist_handler(
            row, col,
            'points',
            PointsArtistHandler(plot_kwargs),
            points
        )

    def _artist_updater(self, obj_type, artist_id, objs):
        def update(i):
            obj = objs[i]
            row, col, artist_handler = self.object_store[obj_type][artist_id]
            if obj is None:
                artist_handler.remove()
                return ()
            if artist_handler.inactive():
                artist_handler.init(self.axs[row][col], obj)
            else:
                artist_handler.update(obj)
            return artist_handler.get_artists()

        return update
    def add_artist_handler(self, row, col, obj_type, artist_handler, objs):
        artist_id = self.generate_id(obj_type)
        self.object_store[obj_type][artist_id] = (row, col, artist_handler)
        self.updaters.append(self._artist_updater(obj_type, artist_id, objs))

    def add_bboxes(self, row, col, bboxes, random_color=True):
        if random_color:
            color = self.get_color('bbox', len(self.object_store['bbox']))
        else:
            color = 'red'
        self.add_artist_handler(
            row, col,
            'bbox',
            BBoxArtistHandler(color),
            bboxes
        )
    def add_labels(self, row, col, ps, color='red', va='bottom', ha='center', formatting='{}', font_size=12):
        self.add_artist_handler(
            row, col,
            'label',
            LabelArtistHandler(color, formatting, ha, va, font_size),
            ps
        )

    def animate(self):
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=list(range(self.frame_num)), 
            blit=True, 
            interval = 1000 / self.fps
        )
        return anim

def plot_images(images, rows, cols, figsize=None):
    """
    Displays a grid of images.
    
    images: a list of images
    rows: the number of rows in the grid
    cols: the number of columns in the grid
    """
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten the array of axes so we can iterate over them easily
    axs = axs.flatten()
    
    # Loop through the images and plot each one
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
    
    plt.show()