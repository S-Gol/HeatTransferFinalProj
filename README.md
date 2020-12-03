# PyFDM
 A simple Finite Difference Method solver for thermal resistance networks. 
 
## Use
 Run using your Python IDE of choice. Requires OpenCV, Numpy, and Plotly
 
 Enter appropriate characteristics for your object(s), material(s), and boundary conditions
 
 Draw the objects in the OpenCV window, using M to toggle materials
 
 Exit the window when done
 
 Draw the boundaries in the OpenCV window, usinb B to toggle materials
 
 Exit the window when done
 
 At this point, the script will convert your input into the appropriate matrix functions and use Numpy's sparse matrix methods to solve the equation. Optionally, change the variable UseGS to "True" to use the slower Gauss-Seidel implementation for demonstration purposes. 
 
 ## Results

![Example drawn input](https://github.com/S-Gol/PyFDM/blob/main/Images/Picture1.png)
![Temperatures](https://github.com/S-Gol/PyFDM/blob/main/Images/Picture2.png)
![Temperature gradient](https://github.com/S-Gol/PyFDM/blob/main/Images/Picture3.png)
![Interior/Exterior convective temperatures](https://github.com/S-Gol/PyFDM/blob/main/Images/Picture4.png)

