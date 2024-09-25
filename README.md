# manualBPPT
This repo shows how to manually implement BPTT for a simple RNN. 
The file RNN_backprop.ipynb shows a one time manual computation of the gradients via BPTT and direct comparison with the PyTorch gradients.
The file RNN_backprop.py shows a 100 step training of two models with SGD optimization -- the first is a model whose backward pass is done via manual BPTT, and the second is a model whose backward pass is done with PyTorch by calling loss.backward() -- and verifies they output the same validation loss.
The file derivations.pdf contains complete derivations of all gradients needed for BPTT.

Idea inspired by Building Makemore Part 4: Becoming a Backprop Ninja (Andrej Karpathy): https://www.youtube.com/watch?v=q8SA3rM6ckI
