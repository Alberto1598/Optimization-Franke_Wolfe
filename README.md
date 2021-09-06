# Optimization-Franke_Wolfe

## Dependencies
The following libraries have been used :
*  `cv2`
* `matplotlib_pyplot`
* `numpy`
* `pandas`
* `tensorflow`

## Instructions

In this project we have implemented four optimization methods for generating adversarial examples for both untargeted and targeted attacks ( FGSM, MI-FGSM, PGD, Franke-Wolfe).

The project should be executed using the following command on a terminal opened in the main directory of the project (inside the code folder): <br />
`python3 main.py [neural_network] [type_of_attack] [modality]` <br/>
where :
* [neural_network] : is the neural network architecture on which is possible to perform an adversarial attack. Three types of convolutional neural network can be selected : "inception_v3", "resnet_v2", "mobilenet_v2".
* [type_of_attack] : is the type of attack it is possible to perform. If you want to make a single attack by using one of the implemented method you can just write "fgsm", "pgd", "mi-fgsm" or "fw-white". Otherwise you can use the option "grid_search" if you want to analyze how each method behaves for different parameters. In the end a plot is produced in order to compare the four methods.
* [modality] : represents the modality of attack. Two types of attack are allowed : "untargeted" and "targeted".
