import os
import vtk
import numpy as np
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from utilities import ReadVTPFile, GetCentroid, ReadVTKFile, WriteVTIFile, WriteVTPFile, ThresholdInBetween, ExtractSurface, LargestConnectedRegion
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class InteractiveEndoSelection():
    def __init__(self):
        self.InputFolder = "/Users/ana/Documents/AnahitaSeresti/05_PrePostCABG/EjectionFractionCalculation/SU24A-CCTA"
        self.slice_base = "Slice_Base.vtp"
        self.slice_apex = "Slice_Apex.vtp"
        self.image_ = "SU24A_CCTA_Registered.vtk"
        self.image = ReadVTKFile(os.path.join(self.InputFolder, self.image_))

    def centerline(self):
        point_base = GetCentroid(ReadVTPFile(os.path.join(self.InputFolder,self.slice_base)))
        point_apex = GetCentroid(ReadVTPFile(os.path.join(self.InputFolder, self.slice_apex)))
        CL_direction = [
            point_base[0] - point_apex[0],
            point_base[1] - point_apex[1],
            point_base[2] - point_apex[2]
                        ]

        CL_direction /= np.linalg.norm(CL_direction)
        np.linalg.norm(CL_direction)

        z_axis = CL_direction
        x_axis = [-z_axis[2], 0, z_axis[0]]
        y_axis = np.cross(x_axis, z_axis)

        center = [
            (point_base[0] + point_apex[0])/2,
            (point_base[1] + point_apex[1])/2,
            (point_base[2] + point_apex[2])/2
        ]
        
        return CL_direction, center

    def SliceWPlane(self, Volume,Origin,Norm):
        plane=vtk.vtkPlane()
        plane.SetOrigin(Origin)
        plane.SetNormal(Norm)
        Slice=vtk.vtkCutter()
        Slice.GenerateTrianglesOff()
        Slice.SetCutFunction(plane)
        Slice.SetInputData(Volume)
        Slice.Update()
        
        return Slice.GetOutput()
    
    def Line(self, point1, point2, res):
        line = vtk.vtkLineSource()
        line.SetPoint1(point1)
        line.SetPoint2(point2)
        line.SetResolution(res)
        line.Update()
        
        return line.GetOutput()
    
    def Rotate(self, rotation, slice, scalar_array_name):
        new_point_x = []
        new_point_y = []

        for i in range(slice.GetNumberOfPoints()):
            point_ = rotation.apply(slice.GetPoint(i))
            new_point_x.append(point_[0])
            new_point_y.append(point_[1])

        new_point_x = np.array(new_point_x)
        new_point_y = np.array(new_point_y)
        scalar_value = vtk_to_numpy(slice.GetPointData().GetArray(scalar_array_name))
        return new_point_x, new_point_y, scalar_value

    def onPress(self, event):
        print(f'coordinate position: {event.xdata}, {event.ydata}')

    def main(self):
        
        CL_direction, center = self.centerline()

        rotation, _ = R.align_vectors([[0, 0, 1]], [CL_direction])
        scalar_array_name = self.image.GetPointData().GetScalars().GetName()

        # Basal point
        point_1 = [
            center[0] + 4.5*CL_direction[0],
            center[1] + 4.5*CL_direction[1],
            center[2] + 4.5*CL_direction[2]
        ]

        # Apical point
        point_2 = [
            center[0] - 4.7*CL_direction[0],
            center[1] - 4.7*CL_direction[1],
            center[2] - 4.7*CL_direction[2]
        ]

        length_LV = [
            point_1[0] - point_2[0],
            point_1[1] - point_2[1],
            point_1[2] - point_2[2]
        ]
        length_LV = np.linalg.norm(length_LV)

        N_Slices = 10

        slice_thickness = length_LV/N_Slices

        LV_CenterLine = self.Line(point_1, point_2, N_Slices)


        # get the slice
        center = LV_CenterLine.GetPoint(5)
        slice_ = self.SliceWPlane(self.image, center, CL_direction)
        BloodPool = ThresholdInBetween(slice_, scalar_array_name, 200, 700)
        BloodPool = ExtractSurface(LargestConnectedRegion(BloodPool))

        bp_centroid = GetCentroid(BloodPool)

        center_rotated = rotation.apply(center)
        centroid_rotated = rotation.apply(bp_centroid)


        new_point_x, new_point_y, scalar_value = self.Rotate(rotation, slice_, scalar_array_name)
        new_point_x_bp, new_point_y_bp, scalar_value_bp = self.Rotate(rotation, BloodPool, scalar_array_name)

        coords_2d = np.column_stack((new_point_x_bp, new_point_y_bp))
        hull = ConvexHull(coords_2d)
        hull_area = hull.volume
        #Area += hull_area

        radius = np.sqrt(hull_area/np.pi)
        #print(hull_area, radius)

        x_circle = []
        y_circle = []
        for theta in np.arange(0,np.pi*2, np.pi/100):
            x_circle.append(centroid_rotated[0] + radius*np.cos(theta))
            y_circle.append(centroid_rotated[1] + radius*np.sin(theta))


        fig, ax = plt.subplots(figsize=(6, 6))

        sc = ax.scatter(new_point_x, new_point_y, c=scalar_value, cmap='gray', s=10, vmin=-1000, vmax=700)
        ax.scatter(x_circle, y_circle, color='green', s = 0.8)

        cursor = Cursor(ax, horizOn= True, vertOn= True, linewidth = 2.0, color = 'red')
        fig.canvas.mpl_connect('button_press_event', self.onPress)
        
        fig.colorbar(sc, label='Scalar Value')
        ax.axis('equal')
        ax.set_xlabel('X (rotated)')
        ax.set_ylabel('Y (rotated)')
        ax.set_xlim([center_rotated[0]-5, center_rotated[0]+5])
        ax.set_ylim([center_rotated[1]-5, center_rotated[1]+5])
        ax.set_title('LV Projection in Short-Axis Plane')
        plt.show()


if __name__ == "__main__":
    InteractiveEndoSelection().main()