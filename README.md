# kalman-project

This repository was made to track my progresses all along the way of my research project. The main purpose here is to create several constructors which will give us methods and tools to filter any signal with Kalman filters. Initially, the Linear Kalman Filter and the Extended Kalman Filter will be the main processing tools to achieve the tasks to fin the optimal estimator. However, 
the creation of an Unscented Kalman Filter class could be a debatable subject.

## Prerequisites

The packages numpy, matplotlib and random must be installed on your machine to be able to run the project, besides any version of Python above 3.x.

## Simulation

The initial ideia given to simulate the filter behavior consists in the distension of a strain gauge, regarding a real instrument proposed from one of my colleagues at UFABC. This strain gauge should be applied in a waistband used to track the motions of a patient during Electrical Impedance Tomography diagnostics, as this analysis requires a high degree of likelihood regarding the model's geometry.

The 'strainGauge.py' class was build up to create a model of this object with the desired math caracteristic. In this case, the measured variable proposed is the resistance of the strain gauge, while the tracked, filtered variable is the radius of curvature of this instrument. The observation equations can be find as functions inside the main file. Additionally, this class is capable to generate random measures to simulate a gaussian acquisition process, a requirement to the Kalman filter application.

As the strain gauge observation equation is not linear, it's essential to make use the Extended Kalman Filter, mainly for its non-linearity processes capability. In this project, it was considered to use the analytical form of the function's first derivative to update the filtering process.

## Deployment
You may run the file main.py and see the graphs supposed to give you the filtered signal of a strain gauge, without deflection. Starting from the "Parameters" file section, you may change the constants used to generate the model.
