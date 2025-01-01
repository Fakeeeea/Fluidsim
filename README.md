# C SPH Fluid Simulation using CUDA

This project implements a fluid simulation engine in C, using wingdi to draw on the window and CUDA to allow for better performance.

## About the Project  

This was a fun yet challenging project to develop. It was my first time diving into CUDA, and it opened up a whole new world of parallel computation and optimization techniques. Being it my first time, it sure has room for improvement.  
A big challenge was actually rendering the particles. This project uses wingdi for the window and the graphics which was not made for high fps projects. I had to manually set the pixels in the bitmap as wingdi's functions were too slow  

The algorithm used, SPH is a particle-based simulation method used to model fluids. Each particle interacts with its neighbors to simulate realistic fluid behavior.  
CUDA allowed the simulation to handle a large number of particles efficiently, allowing also for prettier rendering.  

## Possible future improvements

- Add a parser for a settings and initial particle positions file, to allow the user to change up freely the simulation paramaters without rebuilding the whole project.  
- Seek into some better rendering alternatives, as the current thing I came up with pretty rough.  
- Further optimize the code  

## Extremerly useful resources:
https://www.researchgate.net/publication/221622709_SPH_Based_Shallow_Water_Simulation  
https://matthias-research.github.io/pages/publications/sca03.pdf  
https://lucasschuermann.com/writing/implementing-sph-in-2d  
https://github.com/SebLague/Fluid-Sim  
