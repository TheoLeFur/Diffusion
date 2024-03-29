<h3>About</h3>

Basic implementation of diffusive model for image generation. The idea is to use the Focker-Planck equation, whose solution is the "steepest descent" process in terms of the Wasserstein distance, to introduce noise into the image in the most diffusive manner. The model is then charged to find the initial condition to the stochastic differential equaion, given its trajectory. 

<h3>Setup</h3>
To clone the repository, run

```
git clone https://github.com/TheoLeFur/Diffusion.git
```

Download the requirements using

```
pip install -r requirements.txt
```

To start training the model, run:

```
python3 main.py
```
