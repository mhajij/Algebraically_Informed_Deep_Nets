# Algebraically Informed_Deep Nets (AIDN)

AIDN is a deep learning algorithm to represent any finitely-presented algebraic object with a set of deep neural networks.


Aidn can compute linear and non-linear representations of most finitely-presented algebraic structures such as groups, associative algebras, and Lie algebras.

More details are available here : https://www.researchgate.net/publication/346563873_Algebraically-Informed_Deep_Networks_AIDN_A_Deep_Learning_Approach_to_Represent_Algebraic_Structures


We illusrate AIDN on computing a braid group rep of dimension 2.

To train a braid group representation using AIDN, we start by creating two generator neural networks  f,g :R^2 -> R^2. To train f,g we create an auxiliary neural network for the relations of the braid group. These relations are constrains that the networks f and g must satisfy. We impose these constrains by setting them to a cost function which is then minimized using SGD.

The following groups reps are implemeted in AIDN :

1) The braid group. In particular, AIDN can solve Yang-Baxter equations (over a set or over a vector space).
2) ZxZ.
3) The Symmetric group.

To train a rep for one of the above group you can simply run :

```ruby
main.py -m training -st group_name -dim 4
```
where group_name is braid_group, ZxZ or symmetric_group. Dimension of the disred network should also be specified. Note that the choice of the activation is crucial to determine the type of the representation : linear, affine or non-linear. This argument is optional and it can be customzed using --network_generator_activation.


We have also implemented AIDN to compute reprereation for the Temperley-lieb Algebras.

To train a rep for the Temperly-Lieb algebra use :

```ruby
main.py -m training -st TL -dim 4
```

Additional arguments are also provided for costumized value for the delta constant in the Temperly-Lieb or for generic hyperparameters to train the networks.

Finally, we utilize AIDN to obtain new knot theory invariants using the Reshetikhin-Turaev. The latter can be tested or trained using the file

```ruby
main_DLK_invariants.py
```
Some of the networks we tested are also provided in the repo.

The code is tested on Python 3.7 and tensorflow 1.14.0.
