import glob
import os
import vtk
import argparse



class ConvertPointToLine():
    def __init__(self,Args):
        self.Args = Args

    def main(self):
        filenames = glob.glob(os.path.join(self.Args.InputFolder,"*.vtp"))
        
        for file in filenames:
            centerline = self.ReadVTPFile(file)
            bold_centerline = self.Line(centerline.GetOutput())
            VesselName = os.path.splitext(os.path.basename(file))[0]
            InputFolder = os.path.dirname(file)
            OutputPath = os.path.join(InputFolder, f"Bold_{VesselName}.vtp")
            self.WriteVTPFile(OutputPath, bold_centerline)


    def Line(self, centerline):
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(centerline)
        tube_filter.SetRadius(0.35)  # Adjust this value to change thickness
        tube_filter.SetNumberOfSides(50)  # Higher = smoother tube
        tube_filter.CappingOn()  # Close tube ends
        tube_filter.Update()
        return tube_filter.GetOutput()

    def ReadVTPFile(self, path):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader
    
    def WriteVTPFile(self, FileName, Data):
        writer=vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(FileName)
        writer.SetInputData(Data)
        writer.Update()

if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("-InputFolder", "--InputFolder", dest="InputFolder", type=str, required=True)
    args = Parser.parse_args()

    ConvertPointToLine(args).main()