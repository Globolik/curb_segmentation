import argparse
import copy
import datetime
import os.path
import pathlib
from typing import Tuple, Any
from types import SimpleNamespace

import pylas
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import plyfile
from plyfile import PlyData

from utils.constatnts import \
    RESPONSE_TEMPLATE, ROAD, \
    WINDOW_SIZE, XY_OFFSET, TIME_OFFSET, \
    MAX_ANGLE, MIN_ELEVATION, VERTICAL_DIF,\
    MAX_HORRIZONTAL_DIF, MIN_HORRIZONTAL_DIF

import torch


class BorderExtractor:
    """
    contains functions for road border extraction from Toronto 3d dataset
    """

    def __init__(self):
        if torch.cuda.is_available():
            print('Using GPU for speedup')
            self.device = torch.device('cuda')
        else:
            print('GPU unavailable, process might take up to 30 min')
            self.device = torch.device('cpu')

    def extract_save_border_points(
            self,
            filepath: str,
            export_path: str
    ) -> SimpleNamespace:
        """
        loads point cloud and extracts point coordinates
         that a likely to be a road border
        saves file with point under export dir
        Parameters
        ----------
        filepath : str
            file path to .ply file
        export_path : str
            path to dir or to file with .las

        Returns
        -------
        response : SimpleNamespace
            response with attributes status and message
            if no errors occurred status==200, else 400 and message
            with explanation
        Examples
        --------
        >>> resp = BorderExtractor.extract_save_border_points('1.ply', 'e.las')
        >>> if resp.status==200:
        >>>     print('All good!')
        >>> else:
        >>>     print('Somthing wrong, here is what VVV')
        >>>     print(resp.message)

        """

        # load point cloud
        try:
            angles, labels, points = self.load_point_cloud(filepath)
        except plyfile.PlyParseError as e:
            return self.__prepare_bad_response(str(e))
        except KeyError as e:
            return self.__prepare_bad_response(str(e))
        # extract points of road border
        xyz = self.__border_mask_torch(angles, labels, points)
        # save to a file
        self.__save_coordinates_to_las(
            export_path,
            xyz
        )
        return self.__prepare_good_response()

    def __border_mask_torch(
            self,
            angles,
            labels,
            points
    ) -> Tuple[ndarray, ndarray]:
        """

        Parameters
        ----------
        angles : ndarray
            array of angles from point cloud
        labels : ndarray
            array of classification labels
        points : ndarray
            array with all point cloud data

        Returns
        -------
        xyz : ndarray
            array of points with shape (n, 3) with x y z coordinate of a point
        """
        # offset coordinates
        x, y, z = \
            points['x'] - XY_OFFSET[0],\
            points['y'] - XY_OFFSET[1],\
            points['z']
        labels = np.asarray(labels, dtype='int8')

        # select only road points
        road_mask = labels == ROAD
        x = np.asarray(x, dtype='double')[road_mask]
        y = np.asarray(y, dtype='double')[road_mask]
        z = np.asarray(z, dtype='double')[road_mask]
        angles = angles[road_mask]
        gps_time = np.asarray(
            points['scalar_GPSTime'] - TIME_OFFSET, dtype='double'
        )[road_mask]
        xyz = np.stack([x, y, z], axis=1)

        # write arrays to device (cuda if possible)
        xyz = torch.as_tensor(xyz, dtype=torch.float32).to(self.device)
        angles = torch.Tensor(angles).to(self.device)
        gps_time = torch.as_tensor(
            gps_time * 10e4, dtype=torch.int32
        ).to(self.device)
        sorted_xyz = torch.empty((0, 3), dtype=torch.float32).to(self.device)

        # sort by angle
        unique_angles = torch.unique(angles)
        for an in tqdm(unique_angles,  bar_format='{desc}: {percentage:3.2f}%'):
            angle_mask = angles == an
            xyz_angle = xyz[angle_mask]
            gps_time_sort = gps_time[angle_mask]

            # sort by time
            unique_time = torch.unique(gps_time_sort)
            for time in unique_time:
                time_mask = gps_time_sort == time
                xyz_time = xyz_angle[time_mask]
                # concat to get sorted array
                sorted_xyz = torch.concatenate([sorted_xyz, xyz_time], axis=0)

        # check if points are close on XY plane and far by Z
        mask = self.__distance_check(sorted_xyz)
        # copy back to cpu if was on GPU
        sorted_xyz, mask = sorted_xyz.cpu().numpy(), mask.cpu().numpy()
        # check angle and minimum elevation of neighboring points
        mask = self.__theta_check(sorted_xyz, mask)
        # select border coordinates
        sorted_xyz = sorted_xyz[:-1][mask]
        sorted_xyz[:, 0] += XY_OFFSET[0]
        sorted_xyz[:, 1] += XY_OFFSET[1]
        return sorted_xyz

    def __theta_check(self, xyz: ndarray, mask: ndarray) -> ndarray:
        """
        Find points that have small angle between vectors pointing towards
        neighboring points in a single
        laser beam row. Number of points to use for each side is equal
        to WINDOW_SIZE. If angle is less that MAX_ANGLE than
        check MIN_ELEVATION of the highest neighboring points vectors
        Parameters
        ----------
        xyz : ndarray
            coordinates
        mask : ndarray
            possible border ids

        Returns
        -------
        mask : ndarray
            bool mask for points that passed all checks for border segmentation
        """
        border_id = np.where(mask)[0]
        mask = np.zeros(mask.shape).astype('bool')
        # iterate over all possible border points
        for i in border_id:
            # skip start and end of the array with respect ot WINDOW_SIZE
            if i < WINDOW_SIZE or i > (xyz.shape[0] - WINDOW_SIZE):
                continue
            # get points before selected point
            points_before = xyz[i - WINDOW_SIZE:i]
            # get points after selected point
            points_after = xyz[i + 1:i + WINDOW_SIZE]
            # selected point
            point = xyz[i]
            # compute vectors
            vector_before = \
                self.__compute_neighboring_points_vector(point, points_before)
            vector_after = \
                self.__compute_neighboring_points_vector(point, points_after)
            # split vectors to xy and z components
            vector_before_xy, z_before = vector_before[:2], vector_before[2]
            vector_after_xy, z_after = vector_after[:2], vector_after[2]

            # calculate angle between neighboring points vectors in XY plane
            cos_theta = (np.dot(vector_before_xy, vector_after_xy)) / \
                        (np.linalg.norm(vector_after_xy) *
                         np.linalg.norm(vector_before_xy))
            # angle
            theta = np.arccos(cos_theta) * 180 / np.pi
            # elevation
            distance_z = max(z_before, z_after)
            # if point is suitable save it
            if MAX_ANGLE > theta:
                if distance_z > MIN_ELEVATION:
                    mask[i] = True
        return mask

    @staticmethod
    def load_point_cloud(filepath: str) -> Tuple[ndarray, ndarray, ndarray]:
        """
        load point cloud file of .ply format

        Parameters
        ----------
        filepath : str
            file path to .ply file

        Returns
        -------
        arrays : Tuple[ndarray, ndarray, ndarray]
            arrays with angles, labels, all data
        """
        plydata = PlyData.read(filepath)
        points = plydata['vertex'].data

        angles = points['scalar_ScanAngleRank']
        labels = points['scalar_Label']
        angles = np.asarray(angles, dtype='uint8')
        labels = np.asarray(labels, dtype='uint8')

        return angles, labels, points

    @staticmethod
    def __distance_check(xyz: ndarray) -> ndarray:
        """
        Check two consecutive points are
        close on XY plane and are far on Z axis
        Parameters
        ----------
        xyz : ndarray
            coordinates

        Returns
        -------
        mask : ndarray
            bool mask for points that passed all checks for border segmentation
        """
        # XY plane
        xy = xyz[:, :2]
        horizontal_dif = torch.linalg.norm(xy[:-1] - xy[1:], axis=1)
        # Z axis
        z = xyz[:, 2]
        vertical_dif = torch.abs(z[:-1] - z[1:])

        mask = \
            torch.ge(vertical_dif, VERTICAL_DIF) & \
            torch.lt(horizontal_dif, MAX_HORRIZONTAL_DIF) & \
            torch.ge(horizontal_dif, MIN_HORRIZONTAL_DIF)
        return mask

    @staticmethod
    def __compute_neighboring_points_vector(
            point,
            points
    ) -> Tuple[Any, Any, Any]:
        """

        Parameters
        ----------
        points : ndarray
            points after of before selected $point
        point : ndarray
            x, y, z coordinates of point
        Returns
        -------
        vector : Tuple[float, float, float]
            average of points vector
        """
        return (
            np.sum(points[:, 0] - point[0]) / WINDOW_SIZE,
            np.sum(points[:, 1] - point[1]) / WINDOW_SIZE,
            np.sum(points[:, 2] - point[2]) / WINDOW_SIZE
        )

    @staticmethod
    def __prepare_bad_response(message: str) -> SimpleNamespace:
        """
        create bad response with status 400 and an error message
        Parameters
        ----------
        message : str
            Error message describing the problem
        Returns
        -------
        response : SimpleNamespace
            SimpleNamespace object with status==400 and message==message
        """
        response = copy.deepcopy(RESPONSE_TEMPLATE)
        response.message = message
        return response

    @staticmethod
    def __prepare_good_response() -> SimpleNamespace:
        """
        create good response with status 200
        Parameters
        ----------
        Returns
        -------
        response : SimpleNamespace
            SimpleNamespace object with status==400 and message==message
        """
        response = copy.deepcopy(RESPONSE_TEMPLATE)
        response.status = 200
        return response

    @staticmethod
    def __save_coordinates_to_las(export_path: str, xyz: ndarray) -> None:
        """
        Saves ndarray as .las file. If no path provided saves on Desktop
        Parameters
        ----------
        export_path : str
            folder or file path ending with .las
        xyz : ndarray
            array with coordinates

        Returns
        -------
        """
        if export_path.endswith('.las'):
            os.makedirs(pathlib.Path(export_path).parent, exist_ok=True)
            las = pylas.create()
            las.x = xyz[:, 0]
            las.y = xyz[:, 1]
            las.z = xyz[:, 2]
            print(f'Saving to {export_path}')

            las.write(export_path)
        else:

            desktop_path = \
                os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            if not os.path.exists(desktop_path):
                desktop_path = os.path.join(
                    os.path.join(os.path.expanduser('~')), 'Рабочий стол')
            file_name = \
                'borders_' + \
                str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + \
                '.las'

            print(f'Saving to Desktop. Filename is: {file_name}')
            las = pylas.create()
            las.x = xyz[:, 0]
            las.y = xyz[:, 1]
            las.z = xyz[:, 2]
            las.write(os.path.join(desktop_path, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Road curb cloud segmentation")
    parser.add_argument("input_file",
                        help="path to .ply file")
    parser.add_argument("--output_path",
                        default="",
                        help="path to output .las file")

    args = parser.parse_args()
    extractor = BorderExtractor()
    r = extractor.extract_save_border_points(args.input_file, args.output_path)
    if r.status == 200:
        print('All good, finished')
    else:
        print('Failed to extract curb')
        print(r.message)
