# Splatmesher
Splatmesher is a conversion tool that takes in a Gaussian Splatting file and creates a mesh out of it. The idea is that you can take something that you scanned with a Gaussian Splatting application and convert it to a proper mesh that you can print out on your 3D printer.

## Usage
python splatmesher.py myscan.ply output.obj

## How it is done
The general idea is that each Gaussian is approximated by an ellipsoid. Then it constructs a mesh from merging these ellipsoids.

### Detailed Algorithm
TODO

### Implementation
TODO

### Evaluation
For evaluation of the algorithm, there are some example meshes in the Examples folder which can be rendered from random viewpoints and then compared to rendering the mesh from the same view. The average pixel color difference (L1 difference) is the error to be minimized.
