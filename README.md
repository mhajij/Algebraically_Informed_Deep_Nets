# Algebraically Informed Deep Nets (AIDN)

## Introduction
AIDN is a deep learning algorithm to represent any finitely-presented algebraic object with a set of deep neural networks.

AIDN can compute linear and non-linear representations of most finitely-presented algebraic structures such as groups, associative algebras, and Lie algebras.

More details are available here : https://www.researchgate.net/publication/346563873_Algebraically-Informed_Deep_Networks_AIDN_A_Deep_Learning_Approach_to_Represent_Algebraic_Structures

## Installation

(1) Download the repo to your local drive. 

(2) Create a new conda env using the following command:

```ruby
conda create -n <env_name> --file aidn_conda_env.txt
```

# Getting Started

## Using AIDN to obtain group reps
We illusrate AIDN on computing a braid group rep of dimension 2.

To train a braid group representation using AIDN, we start by creating two generator neural networks ```f,g :R^2 -> R^2 ```. To train f,g we create an auxiliary neural network for the relations of the braid group. These relations are constrains that the networks f and g must satisfy. We impose these constrains by setting them to a cost function which is then minimized using SGD.

```ruby
main.py -m training -st braid_group -dim 2
```


The following groups reps are implemeted in AIDN :

1) The braid group. In particular, AIDN can solve Yang-Baxter equations (over a set or over a vector space).
2) ZxZ.
3) The Symmetric group.

To train a rep for one of the above group you can simply run :

```ruby
main.py -m training -st group_name -dim 4
```
where group_name is braid_group, ZxZ_group or symmetric_group. Dimension of the disred network should also be specified. Note that the choice of the activation is crucial to determine the type of the representation : linear, affine or non-linear. This argument is optional and it can be customized using --network_generator_activation.

## Using AIDN to obtain Temperley-Lieb algebra reps

We have also implemented AIDN to compute representations for the Temperley-Lieb Algebras.

To train a rep for the Temperly-Lieb algebra use :

```ruby
main.py -m training -st TL_algebra -dim 4
```

Additional arguments are also provided for customized value for the delta constant in the Temperly-Lieb or for generic hyperparameters to train the networks.

## Using AIDN to obtain Reshetikhin-Turaev knot invariants


Finally, we utilize AIDN to obtain new knot theory invariants using the Reshetikhin-Turaev construction. The latter can be tested or trained using the file

```ruby
main_knot_invariants.py
```

Some of the networks' weights are provided in the repo.


## Cite
```ruby
@article{hajij2020algebraically,
  title={Algebraically-Informed Deep Networks (AIDN): A Deep Learning Approach to Represent Algebraic Structures},
  author={Hajij, Mustafa and Zamzmi, Ghada and Dawson, Matthew and Muller, Greg},
  journal={arXiv preprint arXiv:2012.01141},
  year={2020}
}
```
