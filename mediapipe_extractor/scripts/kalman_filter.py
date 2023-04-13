import typing as tp

import numpy as np

from filterpy.kalman import KalmanFilter


DELTA_T = 1 / 30
SIGMA_U = 20 * 500
SIGMA_Z = 20
F = [
    [1, DELTA_T],
    [0, 1],
]
H = [
    [1, 0],
]
P = [
    [20 ** 2, 0],
    [0, 0.1 ** 2],
]
R = [[1]]
Q = [
    [1/4 * DELTA_T ** 2, 1/2 * DELTA_T],
    [1/2 * DELTA_T, 1],
]


def init_filter(init_value: tp.Optional[float] = None) -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=0)

    kf.x = np.array([
        [init_value or 0.0],
        [0.0],
    ])
    kf.F = np.array(F)
    kf.H = np.array(H)
    kf.P = np.array(P)
    kf.R = np.array(R) * (SIGMA_Z ** 2)
    kf.Q = np.array(Q) * (DELTA_T ** 2) * (SIGMA_U ** 2)
    return kf


def md_sigma(md: float) -> float:
    return 1 + 1 * md ** 2


class KalmanFilters:
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: np.ndarray
    ) -> None:
        self.width = width
        self.height = height
        self.intrinsic = intrinsic

        self.filters = None

    def init_filters(
        self,
        world_points: np.ndarray,
    ) -> None:
        if world_points.ndim > 1:
            world_points = world_points.reshape(-1)
        self.filters = [init_filter(value) for value in world_points]

    def make_filtering(
        self,
        mp_points: np.ndarray,
        world_points: np.ndarray,
    ) -> np.ndarray:
        if self.filters is None:
            raise Exception('Filters is not initialized!')
        
        if mp_points.ndim > 1:
            mp_points = mp_points.reshape(-1)
        if world_points.ndim > 1:
            world_points = world_points.reshape(-1)

        md: float
        output = np.zeros_like(mp_points)
        for j, new_point in enumerate(world_points[2::3]):
            kf_z = self.filters[3*j+2]
            kf_z.predict()

            z_res = kf_z.residual_of(new_point)[0, 0]
            md = np.sqrt((z_res * z_res) / kf_z.P[0, 0])
            kf_z.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

            if new_point > 50:
                kf_z.update(new_point)

            output[3*j+2] = kf_z.x[0, 0]

            new_x = (mp_points[3*j+0] * self.width - self.principal_x) / self.focal_x * output[3*j+2]
            new_y = (mp_points[3*j+1] * self.height - self.principal_y) / self.focal_y * output[3*j+2]

            kf_x = self.filters[3*j+0]
            kf_y = self.filters[3*j+1]
            kf_x.predict()
            kf_y.predict()

            x_res = kf_x.residual_of(new_x)[0, 0]
            md = np.sqrt((x_res * x_res) / kf_x.P[0, 0])
            kf_x.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

            y_res = kf_y.residual_of(new_y)[0, 0]
            md = np.sqrt((y_res * y_res) / kf_y.P[0, 0])
            kf_y.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

            kf_x.update(new_x)
            kf_y.update(new_y)

            output[3*j+0] = kf_x.x[0, 0]
            output[3*j+1] = kf_y.x[0, 0]
        return output

    @property
    def focal_x(self) -> float:
        return self.intrinsic[0, 0]

    @property
    def focal_y(self) -> float:
        return self.intrinsic[1, 1]

    @property
    def principal_x(self) -> float:
        return self.intrinsic[0, 2]

    @property
    def principal_y(self) -> float:
        return self.intrinsic[1, 2]
