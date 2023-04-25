import numpy as np
from scipy.optimize import least_squares
import cv2

def cost_function(x, *args):
    x_data = args[0]
    y_data = args[1]
    cx = x[0]
    cy = x[1]
    radius = x[2]
    residuals = x_data - (cx + radius * np.cos(y_data))
    cost = np.sum(np.sqrt(1 + (residuals / cauchy_scale)**2) - 1)
    return cost
result = least_squares(cost_function, x0, args=(x_data, y_data))


def cost_function(x):



class camera_path_solver():
    def __init__(self, original_width, original_height,
                      output_width, output_height, first_point=0.5):
        # Check the dimensions
        assert original_width >= output_width
        assert original_height >= output_height

        self.ori_w = original_width
        self.ori_h = original_height
        self.out_w = output_width
        self.out_h = output_height

        # Initialize the problem
        should_solve_x_problem = original_width != output_width
        should_solve_y_problem = original_height != output_height
        self.problem_x, self.problem_y = None, None
        if should_solve_x_problem:
            # make dataset from real-time frames
            self.x0 = [first_point]
            self.problem_x = 
        if should_solve_y_problem:
            self.y0 = [first_point]

    def compute_camera_path(self, point):
        # Solve the problem
        options = {'maxiter': 1000, 'ftol': 1e-9}

        if self.problem_x:
            # data update
            self.x0.append(point)
            
            # 데이터 개수 제한 (지속적으로 프레임을 얻다 보면 데이터의 양이 너무 많아질 수 있음 -> 카메라 경로 연산 증가)
            if (len(self.x0) > 1000):
                self.x0 = self.x0[1:]
            
            res = least_squares(self.problem_x, self.x0, jac='3-point', verbose=0,
                                ftol=options['ftol'], max_nfev=options['maxiter'], loss='cauchy')
        if self.problem_y:
            # data update
            self.y0.append(point)
            
            # 데이터 개수 제한 (지속적으로 프레임을 얻다 보면 데이터의 양이 너무 많아질 수 있음 -> 카메라 경로 연산 증가)
            if (len(self.y0) > 1000):
                self.y0 = self.y0[1:]

            res = least_squares(self.problem_y, self.y0, jac='3-point', verbose=0,
                                ftol=options['ftol'], max_nfev=options['maxiter'], loss='cauchy')
    
        print(self.x0)
        print(res.x)
        return res.x[-1]
        
        
"""
-----------------------------------------------------------
"""




'''
class PolynomialResidual:
    def __init__(self, in_, out_):
        self.in_ = in_
        self.out_ = out_

    def __call__(self, coef):
        a, b, c, d, k = coef
        residual = self.out_ - a * self.in_ - b * self.in_ ** 2 - c * self.in_ ** 3 - d * self.in_ ** 4 - k
        return residual


class CameraPathSolver:
    def __init__(self, original_width, original_height,
                      output_width, output_height):
        # Check the dimensions
        assert original_width >= output_width
        assert original_height >= output_height

        self.ori_w = original_width
        self.ori_h = original_height
        self.out_w = output_width
        self.out_h = output_height

        # Initialize the problem
        should_solve_x_problem = original_width != output_width
        should_solve_y_problem = original_height != output_height
        self.problem_x, self.problem_y = None, None
        if should_solve_x_problem:
            self.problem_x = Problem()
        if should_solve_y_problem:
            self.problem_y = Problem()

    def ComputeCameraPath(self, focus_point_frames, prior_focus_point_frames, original_width, original_height, output_width, output_height):
        assert original_width >= output_width
        assert original_height >= output_height
        should_solve_x_problem = original_width != output_width
        should_solve_y_problem = original_height != output_height
        assert len(focus_point_frames) + len(prior_focus_point_frames) > 0
        
        for i, spf in enumerate(prior_focus_point_frames):
            for sp in spf.point():
                center_x, center_y = sp.norm_point_x(), sp.norm_point_y()
                t = i
                if should_solve_x_problem:
                    self.AddCostFunctionToProblem(t, center_x, problem_x, self.xa_, self.xb_, self.xc_, self.xd_, self.xk_)
                if should_solve_y_problem:
                    self.AddCostFunctionToProblem(t, center_y, problem_y, self.ya_, self.yb_, self.yc_, self.yd_, self.yk_)
        for i, spf in enumerate(focus_point_frames):
            for sp in spf.point():
                center_x, center_y = sp.norm_point_x(), sp.norm_point_y()
                t = i + len(prior_focus_point_frames)
                if should_solve_x_problem:
                    self.AddCostFunction

    # def ComputeDelta(self, in_, original_dimension, output_dimension, a, b, c, d, k):
    #     out = a * in_ + b * in_ ** 2 + c * in_ ** 3 + d * in_ ** 4 + k
    #     delta = (out - 0.5) * original_dimension
    #     max_delta = (original_dimension - output_dimension) / 2.0
    #     delta = np.clip(delta, -max_delta, max_delta)
    #     return delta
    


    def AddCostFunctionToProblem(self, in_, out_, problem, a, b, c, d, k):
        cost_function = PolynomialResidual(in_, out_)
        problem.AddResidualBlock(cost_function, None, [a, b, c, d, k])

    def ComputeCameraPath(self, focus_point_frames, prior_focus_point_frames, original_width, original_height, output_width, output_height):
        assert original_width >= output_width
        assert original_height >= output_height
        should_solve_x_problem = original_width != output_width
        should_solve_y_problem = original_height != output_height
        assert len(focus_point_frames) + len(prior_focus_point_frames) > 0
        
        for i, spf in enumerate(prior_focus_point_frames):
            for sp in spf.point():
                center_x, center_y = sp.norm_point_x(), sp.norm_point_y()
                t = i
                if should_solve_x_problem:
                    self.AddCostFunctionToProblem(t, center_x, problem_x, self.xa_, self.xb_, self.xc_, self.xd_, self.xk_)
                if should_solve_y_problem:
                    self.AddCostFunctionToProblem(t, center_y, problem_y, self.ya_, self.yb_, self.yc_, self.yd_, self.yk_)
        for i, spf in enumerate(focus_point_frames):
            for sp in spf.point():
                center_x, center_y = sp.norm_point_x(), sp.norm_point_y()
                t = i + len(prior_focus_point_frames)
                if should_solve_x_problem:
                    self.AddCostFunction

'''