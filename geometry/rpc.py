import numpy as np
import os
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _affine_from_3pts(corners: np.ndarray, offset_corners: np.ndarray) -> np.ndarray:
    """计算 2x3 仿射矩阵。

    优先使用 OpenCV；若环境无 cv2，则回退到 numpy 线性求解，保持工程可运行性。
    """
    if cv2 is not None:
        return cv2.getAffineTransform(corners, offset_corners)
    a = np.concatenate([corners, np.ones((3, 1), dtype=corners.dtype)], axis=1)  # [3,3]
    x = np.linalg.solve(a, offset_corners[:, 0])
    y = np.linalg.solve(a, offset_corners[:, 1])
    return np.stack([x, y], axis=0)


class RPCModelParameterTorch:
    def __init__(self, data=torch.zeros(170, dtype=torch.double)):
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]

        self.Clear_Adjust()

        self.device = self.LINE_OFF.device
        self.logger = None

    def load_from_file(self, filepath):
        """
        Read direct RPC from file and calculate inverse RPC if absent.
        """
        if os.path.exists(filepath) is False:
            self.log("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()
            rfm_line = -1
            for line, text in enumerate(all_the_text):
                if "RFM_CORRECTION_PARAMETERS" in text:
                    rfm_line = line
                    break

        data = [np.float64(text.split()[1]) for text in (all_the_text[:rfm_line - 1] if rfm_line > 0 else all_the_text)]
        data = torch.from_numpy(np.array(data, dtype=np.float64)).to(torch.double)

        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]
        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        if data.shape[0] >= 170:
            self.LATNUM = data[90:110]
            self.LATDEM = data[110:130]
            self.LONNUM = data[130:150]
            self.LONDEM = data[150:170]
        else:
            self.Calculate_Inverse_RPC()

        if rfm_line > 0:
            self.raw_adjust_params = [np.float64(text.split()[1]) for text in all_the_text[rfm_line + 1:]]
            self.Calculate_Adjust()
        else:
            self.raw_adjust_params = None

    def Create_Virtual_3D_Grid(self, xy_sample=30, z_sample=20):
        lat_max = self.LAT_OFF + self.LAT_SCALE
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LONG_OFF + self.LONG_SCALE
        lon_min = self.LONG_OFF - self.LONG_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        lat = torch.linspace(lat_min, lat_max, xy_sample).to(self.device, dtype=torch.double)
        lon = torch.linspace(lon_min, lon_max, xy_sample).to(self.device, dtype=torch.double)
        hei = torch.linspace(hei_min, hei_max, z_sample).to(self.device, dtype=torch.double)

        lat, lon, hei = torch.meshgrid(lat, lon, hei, indexing='ij')

        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp, line = self.RPC_OBJ2PHOTO(lat, lon, hei)
        grid = torch.stack((samp, line, lat, lon, hei), dim=-1).to(self.device, dtype=torch.double)

        selected_grid = []
        for g in grid:
            flag = [g[0] < samp_min, g[0] > samp_max, g[1] < line_min, g[1] > line_max]
            if True in flag:
                continue
            else:
                selected_grid.append(g)

        if len(selected_grid) > 0:
            grid = torch.stack(selected_grid, dim=0).to(self.device, dtype=torch.double)
        else:
            self.log("警告: 虚拟格网点均在影像范围外。")
            grid = torch.empty(0, 5, device=self.device, dtype=torch.double)

        return grid

    def _solve_lstsq(self, ma, lv, x=None, k=1):
        assert ma.shape[0] == ma.shape[1], "ma with shape {} is not a square matrix.".format(ma.shape[0], ma.shape[1])

        if x is None:
            x = torch.zeros(ma.shape[0], dtype=torch.double, device=self.device)

        n = ma.shape[0]
        mak = ma.clone()
        mak += k * torch.eye(n).to(self.device, dtype=torch.double)
        lk = lv.clone()

        finish_time = 0

        for times in range(1000):
            try:
                x1 = torch.linalg.solve(mak, lk)
            except torch.linalg.LinAlgError:
                self.log("警告: 最小二乘解算中矩阵奇异，增加k值。")
                k *= 10
                mak = ma.clone() + k * torch.eye(n).to(self.device, dtype=torch.double)
                continue

            dif = torch.abs(x1 - x)
            maxdif = torch.max(dif)
            x = x1
            lk = lv + k * x

            finish_time = times + 1
            if maxdif < 1.0e-10:
                break
        return x, finish_time

    def Solve_Inverse_RPC(self, grid):
        samp, line, lat, lon, hei = torch.hsplit(grid, 5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp = samp - self.SAMP_OFF
        samp = samp / self.SAMP_SCALE
        line = line - self.LINE_OFF
        line = line / self.LINE_SCALE

        lat = lat - self.LAT_OFF
        lat = lat / self.LAT_SCALE
        lon = lon - self.LONG_OFF
        lon = lon / self.LONG_SCALE
        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp, line, hei)

        n_num = coef.shape[0]
        A = torch.zeros((n_num * 2, 78)).to(self.device, dtype=torch.double)
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        l = torch.cat((lat, lon), -1)
        l = -l

        ATA = torch.matmul(A.T, A)
        ATl = torch.matmul(A.T, l)

        x, times = self._solve_lstsq(ATA, ATl)

        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]

        return times

    def Calculate_Inverse_RPC(self):
        grid = self.Create_Virtual_3D_Grid()
        if grid.shape[0] == 0:
            self.log("错误: 无法创建虚拟格网，反向RPC计算失败。")
            return -1
        times = self.Solve_Inverse_RPC(grid)
        return times

    def Inverse_Adjust(self):
        R = self.adjust_params[:, :2]
        t = self.adjust_params[:, 2]
        try:
            R_inv = torch.inverse(R)
            t_new = -(R_inv @ t)
            self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)
        except torch.linalg.LinAlgError:
            raise ValueError(f"仿射矩阵：\n{self.adjust_params.detach().cpu().numpy()}\n不可逆")

    def Clear_Adjust(self):
        self.adjust_params = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=torch.double, device=self.LNUM.device)
        self.Inverse_Adjust()

    def Update_Adjust(self, new_adjust_params: torch.Tensor):
        if isinstance(new_adjust_params, np.ndarray):
            new_adjust_params = torch.from_numpy(new_adjust_params)
        new_adjust_params = new_adjust_params.to(self.adjust_params.device).to(torch.double)

        def merge_adjust(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            device = A.device
            dtype = A.dtype

            bottom_row = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)

            A_h = torch.cat([A, bottom_row], dim=0)
            B_h = torch.cat([B, bottom_row], dim=0)

            C_h = B_h @ A_h
            return C_h[:2, :]

        self.adjust_params = merge_adjust(self.adjust_params, new_adjust_params).to(self.adjust_params.device).to(torch.double)
        self.Inverse_Adjust()

    def Calculate_Adjust(self):
        corners = np.array([[0., 0.], [100., 0.], [0., 100.]], dtype=np.float32)  # line samp
        offset_line = self.raw_adjust_params[0] + self.raw_adjust_params[1] * corners[:, 1] + self.raw_adjust_params[2] * corners[:, 0]
        offset_samp = self.raw_adjust_params[3] + self.raw_adjust_params[4] * corners[:, 1] + self.raw_adjust_params[5] * corners[:, 0]

        offset_corners = corners - np.stack([offset_line, offset_samp], axis=1)

        af_trans = _affine_from_3pts(corners, offset_corners)
        self.Update_Adjust(torch.from_numpy(af_trans))

    def Merge_Adjust(self):
        identity_adjust = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=torch.double, device=self.device)

        if torch.allclose(self.adjust_params, identity_adjust, atol=1e-8):
            self.log("Adjust parameters are already identity. No merge needed.")
            return

        grid_obj = self.Create_Virtual_3D_Grid(xy_sample=50, z_sample=30)

        if grid_obj.shape[0] == 0:
            self.log("错误: 无法创建用于合并的虚拟格网。操作中止。")
            return

        samp_target = grid_obj[:, 0]
        line_target = grid_obj[:, 1]
        lat = grid_obj[:, 2]
        lon = grid_obj[:, 3]
        hei = grid_obj[:, 4]

        P = (lat - self.LAT_OFF) / self.LAT_SCALE
        L = (lon - self.LONG_OFF) / self.LONG_SCALE
        H = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        line_target_norm = (line_target - self.LINE_OFF) / self.LINE_SCALE
        samp_target_norm = (samp_target - self.SAMP_OFF) / self.SAMP_SCALE

        coef = self.RPC_PLH_COEF(P, L, H)
        n_num = coef.shape[0]

        A_L = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_L[:, 0:20] = coef
        A_L[:, 20:39] = -line_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_L = line_target_norm * coef[:, 0]

        ATA_L = A_L.T @ A_L
        ATl_L = A_L.T @ l_L

        A_S = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_S[:, 0:20] = coef
        A_S[:, 20:39] = -samp_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_S = samp_target_norm * coef[:, 0]

        ATA_S = A_S.T @ A_S
        ATl_S = A_S.T @ l_S

        x_L, _ = self._solve_lstsq(ATA_L, ATl_L)
        x_S, _ = self._solve_lstsq(ATA_S, ATl_S)

        self.LNUM = x_L[0:20].clone()
        self.LDEM[0] = 1.0
        self.LDEM[1:20] = x_L[20:39].clone()
        self.SNUM = x_S[0:20].clone()
        self.SDEM[0] = 1.0
        self.SDEM[1:20] = x_S[20:39].clone()

        self.Clear_Adjust()
        self.Calculate_Inverse_RPC()

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = torch.zeros((n_num, 20), dtype=torch.double, device=P.device)
        coef[:, 0] = 1.0
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * L * H
        coef[:, 11] = L * L * L
        coef[:, 12] = L * P * P
        coef[:, 13] = L * H * H
        coef[:, 14] = L * L * P
        coef[:, 15] = P * P * P
        coef[:, 16] = P * H * H
        coef[:, 17] = L * L * H
        coef[:, 18] = P * P * H
        coef[:, 19] = H * H * H
        return coef

    def convert_tensor(self, arr, device):
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.double, device=device)
        else:
            return torch.as_tensor(arr, dtype=torch.double, device=device)

    # ============================================================
    # Core helpers
    # ============================================================

    def _photo2obj_from_origcoords(self, samp_orig, line_orig, hei):
        """
        Inverse RPC without affine adjust.
        Inputs are already in the original RPC image coordinate system.
        """
        hei = self.convert_tensor(hei, self.device)
        samp_orig = self.convert_tensor(samp_orig, self.device)
        line_orig = self.convert_tensor(line_orig, self.device)

        samp_norm = (samp_orig - self.SAMP_OFF) / self.SAMP_SCALE
        line_norm = (line_orig - self.LINE_OFF) / self.LINE_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp_norm, line_norm, hei_norm)

        lat_norm = torch.sum(coef * self.LATNUM, dim=-1) / torch.sum(coef * self.LATDEM, dim=-1)
        lon_norm = torch.sum(coef * self.LONNUM, dim=-1) / torch.sum(coef * self.LONDEM, dim=-1)

        lat = lat_norm * self.LAT_SCALE + self.LAT_OFF
        lon = lon_norm * self.LONG_SCALE + self.LONG_OFF
        return lat, lon

    def _photo2obj_with_adjust(self, insamp, inline, inhei, adjust_params: torch.Tensor):
        """
        Differentiable helper:
        From final observed (samp, line, hei) to (lat, lon)
        using an explicitly supplied affine adjust matrix.

        adjust_params: (2, 3), mapping final (line, samp) -> original RPC image coordinates
        """
        hei = self.convert_tensor(inhei, self.device)
        samp = self.convert_tensor(insamp, self.device)
        line = self.convert_tensor(inline, self.device)
        adjust_params = self.convert_tensor(adjust_params, self.device).reshape(2, 3)

        transformed_points = torch.stack([line, samp], dim=-1) @ adjust_params[:, :2].T + adjust_params[:, 2]
        line_orig = transformed_points[:, 0]
        samp_orig = transformed_points[:, 1]

        return self._photo2obj_from_origcoords(samp_orig, line_orig, hei)

    def _linesamp2xy_from_origcoords(self, line_orig, samp_orig, h, xy_center=None, xy_scale=None):
        """
        Differentiable helper:
        From original RPC image coordinates (line_orig, samp_orig, h) -> (x, y)
        without affine adjust.
        """
        lat, lon = self._photo2obj_from_origcoords(samp_orig, line_orig, h)
        yx = self.latlon2yx(torch.stack([lat, lon], dim=-1), center=xy_center, scale=xy_scale)
        y = yx[:, 0]
        x = yx[:, 1]
        return x, y

    def _linesamp2xy_with_adjust(self, line, samp, h, adjust_params: torch.Tensor, xy_center=None, xy_scale=None):
        """
        Differentiable helper:
        From final observed (line, samp, h) -> (x, y) with supplied affine adjust.
        """
        lat, lon = self._photo2obj_with_adjust(samp, line, h, adjust_params)
        yx = self.latlon2yx(torch.stack([lat, lon], dim=-1), center=xy_center, scale=xy_scale)
        y = yx[:, 0]
        x = yx[:, 1]
        return x, y

    # ============================================================
    # Geometry features
    # ============================================================

    def compute_geometry_features(self, Coords: torch.Tensor, dem: torch.Tensor = None, xy_center=None, xy_scale=None) -> torch.Tensor:
        """
        For each input pixel coordinate (line, samp), compute a 20D geometry feature:
            1. dX/dh, d2X/dh2, dY/dh, d2Y/dh2                                 (4)
            2. dX/dA_00, dX/dA_01, dX/dA_02, dX/dA_10, dX/dA_11, dX/dA_12,   (12)
               dY/dA_00, dY/dA_01, dY/dA_02, dY/dA_10, dY/dA_11, dY/dA_12
            3. dX/dline, dX/dsamp, dY/dline, dY/dsamp                         (4)

        Args:
            Coords: (N, 2), each row is (line, samp)
            dem:    (N,), optional; if None, zeros will be used
            xy_center: optional normalization center in (y, x) order
            xy_scale: optional normalization scale in (y, x) order

        Returns:
            features: (N, 20), torch.double
        """
        coords = self.convert_tensor(Coords, self.device)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Coords must have shape (N, 2), got {tuple(coords.shape)}")

        n = coords.shape[0]
        if dem is None:
            h = torch.zeros(n, dtype=torch.double, device=self.device)
        else:
            h = self.convert_tensor(dem, self.device).reshape(-1)
            if h.shape[0] != n:
                raise ValueError(f"dem must have shape (N,), got {tuple(h.shape)} for N={n}")

        # external observed pixel coordinates
        line = coords[:, 0].clone().detach().requires_grad_(True)
        samp = coords[:, 1].clone().detach().requires_grad_(True)
        h = h.clone().detach().requires_grad_(True)

        A = self.adjust_params.clone().detach().to(dtype=torch.double, device=self.device)

        # ------------------------------------------------------------------
        # Full observed path: (line, samp, h) --A--> (line_orig, samp_orig) --RPC--> (X, Y)
        # ------------------------------------------------------------------
        x, y = self._linesamp2xy_with_adjust(line, samp, h, A, xy_center=xy_center, xy_scale=xy_scale)

        ones_x = torch.ones_like(x)
        ones_y = torch.ones_like(y)

        # 1) h derivatives
        dx_dh = torch.autograd.grad(x, h, grad_outputs=ones_x, create_graph=True, retain_graph=True)[0]
        dy_dh = torch.autograd.grad(y, h, grad_outputs=ones_y, create_graph=True, retain_graph=True)[0]

        d2x_dh2 = torch.autograd.grad(dx_dh, h, grad_outputs=torch.ones_like(dx_dh), create_graph=False, retain_graph=True)[0]
        d2y_dh2 = torch.autograd.grad(dy_dh, h, grad_outputs=torch.ones_like(dy_dh), create_graph=False, retain_graph=True)[0]

        # 2) derivatives wrt observed line/samp
        dx_dline = torch.autograd.grad(x, line, grad_outputs=ones_x, create_graph=False, retain_graph=True)[0]
        dx_dsamp = torch.autograd.grad(x, samp, grad_outputs=ones_x, create_graph=False, retain_graph=True)[0]
        dy_dline = torch.autograd.grad(y, line, grad_outputs=ones_y, create_graph=False, retain_graph=True)[0]
        dy_dsamp = torch.autograd.grad(y, samp, grad_outputs=ones_y, create_graph=False, retain_graph=True)[0]

        # ------------------------------------------------------------------
        # 3) affine derivatives (corrected implementation)
        #
        # Since:
        #   line_orig = A00*line + A01*samp + A02
        #   samp_orig = A10*line + A11*samp + A12
        #
        # We compute:
        #   dX/dline_orig, dX/dsamp_orig, dY/dline_orig, dY/dsamp_orig
        # then apply chain rule exactly:
        #
        #   dX/dA00 = dX/dline_orig * line
        #   dX/dA01 = dX/dline_orig * samp
        #   dX/dA02 = dX/dline_orig
        #   dX/dA10 = dX/dsamp_orig * line
        #   dX/dA11 = dX/dsamp_orig * samp
        #   dX/dA12 = dX/dsamp_orig
        # and same for Y.
        # ------------------------------------------------------------------
        line_const = line.detach()
        samp_const = samp.detach()
        h_const = h.detach()

        line_orig = (A[0, 0] * line_const + A[0, 1] * samp_const + A[0, 2]).clone().detach().requires_grad_(True)
        samp_orig = (A[1, 0] * line_const + A[1, 1] * samp_const + A[1, 2]).clone().detach().requires_grad_(True)

        x_orig, y_orig = self._linesamp2xy_from_origcoords(
            line_orig, samp_orig, h_const, xy_center=xy_center, xy_scale=xy_scale
        )

        dx_dline_orig = torch.autograd.grad(x_orig, line_orig, grad_outputs=torch.ones_like(x_orig), create_graph=False, retain_graph=True)[0]
        dx_dsamp_orig = torch.autograd.grad(x_orig, samp_orig, grad_outputs=torch.ones_like(x_orig), create_graph=False, retain_graph=True)[0]
        dy_dline_orig = torch.autograd.grad(y_orig, line_orig, grad_outputs=torch.ones_like(y_orig), create_graph=False, retain_graph=True)[0]
        dy_dsamp_orig = torch.autograd.grad(y_orig, samp_orig, grad_outputs=torch.ones_like(y_orig), create_graph=False, retain_graph=True)[0]

        # X wrt A
        dX_dA00 = dx_dline_orig * line_const
        dX_dA01 = dx_dline_orig * samp_const
        dX_dA02 = dx_dline_orig
        dX_dA10 = dx_dsamp_orig * line_const
        dX_dA11 = dx_dsamp_orig * samp_const
        dX_dA12 = dx_dsamp_orig

        # Y wrt A
        dY_dA00 = dy_dline_orig * line_const
        dY_dA01 = dy_dline_orig * samp_const
        dY_dA02 = dy_dline_orig
        dY_dA10 = dy_dsamp_orig * line_const
        dY_dA11 = dy_dsamp_orig * samp_const
        dY_dA12 = dy_dsamp_orig

        features = torch.stack([
            dx_dh,
            d2x_dh2,
            dy_dh,
            d2y_dh2,

            dX_dA00, dX_dA01, dX_dA02, dX_dA10, dX_dA11, dX_dA12,
            dY_dA00, dY_dA01, dY_dA02, dY_dA10, dY_dA11, dY_dA12,

            dx_dline,
            dx_dsamp,
            dy_dline,
            dy_dsamp,
        ], dim=-1)

        return features

    # ============================================================
    # Public RPC interfaces
    # ============================================================

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei, output_type='tensor'):
        """
        From (lat, lon, hei) to (samp, line) using the direct RPC.
        """
        lat = self.convert_tensor(inlat, self.device)
        lon = self.convert_tensor(inlon, self.device)
        hei = self.convert_tensor(inhei, self.device)

        is_batched = lat.dim() > 0
        if not is_batched:
            lat = lat.unsqueeze(0)
            lon = lon.unsqueeze(0)
            hei = hei.unsqueeze(0)

        lat_norm = (lat - self.LAT_OFF) / self.LAT_SCALE
        lon_norm = (lon - self.LONG_OFF) / self.LONG_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat_norm, lon_norm, hei_norm)

        samp_norm = torch.sum(coef * self.SNUM, dim=-1) / torch.sum(coef * self.SDEM, dim=-1)
        line_norm = torch.sum(coef * self.LNUM, dim=-1) / torch.sum(coef * self.LDEM, dim=-1)

        samp = samp_norm * self.SAMP_SCALE + self.SAMP_OFF
        line = line_norm * self.LINE_SCALE + self.LINE_OFF

        transformed_points = torch.stack([line, samp], dim=-1) @ self.adjust_params_inv[:, :2].T + self.adjust_params_inv[:, 2]
        line_final = transformed_points[:, 0]
        samp_final = transformed_points[:, 1]

        if not is_batched:
            line_final = line_final.squeeze(0)
            samp_final = samp_final.squeeze(0)

        if output_type == 'numpy':
            samp_final = samp_final.cpu().numpy()
            line_final = line_final.cpu().numpy()

        return samp_final, line_final

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei, output_type='tensor'):
        """
        From (samp, line, hei) to (lat, lon) using the inverse RPC.
        """
        hei = self.convert_tensor(inhei, self.device)
        samp = self.convert_tensor(insamp, self.device)
        line = self.convert_tensor(inline, self.device)

        is_batched = samp.dim() > 0
        if not is_batched:
            samp = samp.unsqueeze(0)
            line = line.unsqueeze(0)
            hei = hei.unsqueeze(0)

        transformed_points = torch.stack([line, samp], dim=-1) @ self.adjust_params[:, :2].T + self.adjust_params[:, 2]
        line_orig = transformed_points[:, 0]
        samp_orig = transformed_points[:, 1]

        lat, lon = self._photo2obj_from_origcoords(samp_orig, line_orig, hei)

        if not is_batched:
            lat = lat.squeeze(0)
            lon = lon.squeeze(0)

        if output_type == 'numpy':
            lon = lon.cpu().numpy()
            lat = lat.cpu().numpy()

        return lat, lon

    # ============================================================
    # Web Mercator normalization helpers
    # ============================================================

    def _prepare_yx_center_scale(self, center=None, scale=None):
        center_t = None
        scale_t = None

        if center is not None:
            center_t = self.convert_tensor(center, self.device).reshape(-1)
            if center_t.numel() != 2:
                raise ValueError(f"center must be None or have 2 elements in (y, x) order, got shape {tuple(center_t.shape)}")

        if scale is not None:
            scale_t = self.convert_tensor(scale, self.device).reshape(-1)
            if scale_t.numel() == 1:
                scale_t = scale_t.repeat(2)
            elif scale_t.numel() != 2:
                raise ValueError(f"scale must be None, a scalar, or have 2 elements in (y, x) order, got shape {tuple(scale_t.shape)}")
            if torch.any(scale_t == 0):
                raise ValueError("scale must be non-zero")

        return center_t, scale_t

    def _apply_yx_normalization(self, yx: torch.Tensor, center=None, scale=None) -> torch.Tensor:
        center_t, scale_t = self._prepare_yx_center_scale(center, scale)
        if center_t is not None:
            yx = yx - center_t.unsqueeze(0)
        if scale_t is not None:
            yx = yx / scale_t.unsqueeze(0)
        return yx

    def _undo_yx_normalization(self, yx: torch.Tensor, center=None, scale=None) -> torch.Tensor:
        center_t, scale_t = self._prepare_yx_center_scale(center, scale)
        if scale_t is not None:
            yx = yx * scale_t.unsqueeze(0)
        if center_t is not None:
            yx = yx + center_t.unsqueeze(0)
        return yx

    def latlon2yx(self, latlon: torch.Tensor, center=None, scale=None):
        """
        (lat, lon) -> (y, x), shape (N,2)
        Optional normalization:
            yx_norm = (yx - center) / scale
        center / scale are in (y, x) order
        """
        latlon = self.convert_tensor(latlon, self.device)
        r = 6378137.0
        lon_rad = latlon[:, 1] * torch.pi / 180.0
        lat_rad = latlon[:, 0] * torch.pi / 180.0
        x = r * lon_rad
        y = r * torch.log(torch.tan(torch.pi / 4.0 + lat_rad / 2.0))
        yx = torch.stack([y, x], dim=-1)
        return self._apply_yx_normalization(yx, center=center, scale=scale)

    def yx2latlon(self, yx: torch.Tensor, center=None, scale=None):
        """
        (y, x) -> (lat, lon), shape (N,2)
        If normalized center/scale are provided, they are inverted first.
        """
        yx = self.convert_tensor(yx, self.device)
        yx = self._undo_yx_normalization(yx, center=center, scale=scale)
        r = 6378137.0
        lon = (180.0 * yx[:, 1]) / (torch.pi * r)
        lat = (2.0 * torch.atan(torch.exp(yx[:, 0] / r)) - torch.pi * 0.5) * 180.0 / torch.pi
        return torch.stack([lat, lon], dim=-1)

    def RPC_XY2LINESAMP(self, x_in, y_in, h_in, output_type='tensor', xy_center=None, xy_scale=None):
        x = self.convert_tensor(x_in, self.device)
        y = self.convert_tensor(y_in, self.device)
        h = self.convert_tensor(h_in, self.device)

        is_batched = x.dim() > 0
        if not is_batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            h = h.unsqueeze(0)

        latlon = self.yx2latlon(torch.stack([y, x], dim=-1), center=xy_center, scale=xy_scale)
        samp, line = self.RPC_OBJ2PHOTO(latlon[:, 0], latlon[:, 1], h)

        if not is_batched:
            line = line.squeeze(0)
            samp = samp.squeeze(0)

        if output_type == 'numpy':
            line = line.cpu().numpy()
            samp = samp.cpu().numpy()

        return line, samp

    def RPC_LINESAMP2XY(self, line_in, samp_in, h_in, output_type='tensor', xy_center=None, xy_scale=None):
        line = self.convert_tensor(line_in, self.device)
        samp = self.convert_tensor(samp_in, self.device)
        h = self.convert_tensor(h_in, self.device)

        is_batched = line.dim() > 0
        if not is_batched:
            line = line.unsqueeze(0)
            samp = samp.unsqueeze(0)
            h = h.unsqueeze(0)

        lat, lon = self.RPC_PHOTO2OBJ(samp, line, h)
        yx = self.latlon2yx(torch.stack([lat, lon], dim=-1), center=xy_center, scale=xy_scale)
        y, x = yx[:, 0], yx[:, 1]

        if not is_batched:
            x = x.squeeze(0)
            y = y.squeeze(0)

        if output_type == 'numpy':
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        return x, y

    # ============================================================
    # Device / save / logger
    # ============================================================

    def to_gpu(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                self.log("CUDA not available, using CPU.")
                device = 'cpu'

        self.device = torch.device(device)

        self.LINE_OFF = self.LINE_OFF.to(self.device)
        self.SAMP_OFF = self.SAMP_OFF.to(self.device)
        self.LAT_OFF = self.LAT_OFF.to(self.device)
        self.LONG_OFF = self.LONG_OFF.to(self.device)
        self.HEIGHT_OFF = self.HEIGHT_OFF.to(self.device)
        self.LINE_SCALE = self.LINE_SCALE.to(self.device)
        self.SAMP_SCALE = self.SAMP_SCALE.to(self.device)
        self.LAT_SCALE = self.LAT_SCALE.to(self.device)
        self.LONG_SCALE = self.LONG_SCALE.to(self.device)
        self.HEIGHT_SCALE = self.HEIGHT_SCALE.to(self.device)

        self.LNUM = self.LNUM.to(self.device)
        self.LDEM = self.LDEM.to(self.device)
        self.SNUM = self.SNUM.to(self.device)
        self.SDEM = self.SDEM.to(self.device)

        self.LATNUM = self.LATNUM.to(self.device)
        self.LATDEM = self.LATDEM.to(self.device)
        self.LONNUM = self.LONNUM.to(self.device)
        self.LONDEM = self.LONDEM.to(self.device)

        self.adjust_params = self.adjust_params.to(self.device)
        self.adjust_params_inv = self.adjust_params_inv.to(self.device)

    def save_rpc_to_file(self, filepath):
        original_device = self.device
        if self.device.type != 'cpu':
            self.to_gpu('cpu')

        addition0 = [
            'LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:',
            'LINE_SCALE:', 'SAMP_SCALE:', 'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:',
            'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:', 'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:',
            'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:', 'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:',
            'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:', 'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:',
            'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:', 'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:',
            'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:', 'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:',
            'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:', 'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:',
            'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:', 'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:',
            'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:', 'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:',
            'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:', 'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:',
            'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:', 'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:',
            'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:', 'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:',
            'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:', 'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:',
            'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:', 'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:',
            'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:', 'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:',
            'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:', 'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:',
            'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:', 'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:',
            'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:', 'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:',
            'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:', 'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:',
            'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:', 'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:',
            'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:', 'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:',
            'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:', 'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:',
            'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:', 'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:',
            'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:', 'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:',
            'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:', 'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:',
            'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:', 'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:',
            'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:', 'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:',
            'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:', 'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:',
            'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:', 'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:',
            'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:', 'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:',
            'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:', 'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:',
            'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:', 'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:',
            'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:', 'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:',
            'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:', 'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:',
            'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:', 'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:',
            'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:', 'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:',
            'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:', 'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:',
            'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:', 'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:',
            'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:', 'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:',
            'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:', 'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:',
            'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:', 'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:'
        ]
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees', 'meters']

        text = ""
        text += addition0[0] + " " + str(self.LINE_OFF.item()) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF.item()) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF.item()) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF.item()) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF.item()) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE.item()) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE.item()) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE.item()) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE.item()) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE.item()) + " " + addition1[9] + "\n"

        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10].item()) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 30].item()) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50].item()) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70].item()) + "\n"
        for i in range(90, 110):
            text += addition0[i] + " " + str(self.LATNUM[i - 90].item()) + "\n"
        for i in range(110, 130):
            text += addition0[i] + " " + str(self.LATDEM[i - 110].item()) + "\n"
        for i in range(130, 150):
            text += addition0[i] + " " + str(self.LONNUM[i - 130].item()) + "\n"
        for i in range(150, 170):
            text += addition0[i] + " " + str(self.LONDEM[i - 150].item()) + "\n"

        with open(filepath, "w") as f:
            f.write(text)

        if original_device.type != 'cpu':
            self.to_gpu(original_device)

    def set_logger(self, logger):
        self.logger = logger

    def log(self, msg):
        if self.logger is None:
            print(msg)
        else:
            self.logger(msg)


def load_rpc(rpc_path: str, to_gpu=False) -> RPCModelParameterTorch:
    rpc = RPCModelParameterTorch()
    rpc.load_from_file(rpc_path)
    if to_gpu:
        rpc.to_gpu()
    return rpc


def project_linesamp(rpc1: RPCModelParameterTorch, rpc2: RPCModelParameterTorch, lines, samps, heights, output_type='tensor'):
    """
    return : (lines, samps)
    """
    lat, lon = rpc1.RPC_PHOTO2OBJ(samps, lines, heights)
    samps, lines = rpc2.RPC_OBJ2PHOTO(lat, lon, heights, output_type)
    return lines, samps
