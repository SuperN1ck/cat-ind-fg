# Category-Independent Articulated Object Tracking with Factor Graphs
[Nick Heppert](https://rl.uni-freiburg.de/people/heppert), [Toki Migimatsu](https://cs.stanford.edu/~takatoki/), [Brent Yi](https://brentyi.com/), [Claire Chen](https://msl.stanford.edu/people/clairechen), [Jeannette Bohg](https://web.stanford.edu/~bohg/)

[**arXiv**](https://arxiv.org/abs/2205.03721) | [**website**](https://tinyurl.com/ycyva37v) |
*Presented at IROS 20222, Kyoto, Japan*

<p align="center">
  <img src="assets/overview.png" alt="Overview" width="500" />
</p>

## 📔 Abstract
Robots deployed in human-centric environments may need to manipulate a diverse range of articulated objects, such as doors, dishwashers, and cabinets. Articulated objects often come with unexpected articulation mechanisms that are inconsistent with categorical priors: for example, a drawer might rotate about a hinge joint instead of sliding open. We propose a category-independent framework for predicting the articulation models of unknown objects from sequences of RGB-D images. The prediction is performed by a two-step process: first, a visual perception module tracks object part poses from raw images, and second, a factor graph takes these poses and infers the articulation model including the current configuration between the parts as a 6D twist. We also propose a manipulation-oriented metric to evaluate predicted joint twists in terms of how well a compliant robot controller would be able to manipulate the articulated object given the predicted twist. We demonstrate that our visual perception and factor graph modules outperform baselines on simulated data and show the applicability of our factor graph on real world data.

## 👨‍💻 Code Release
*This repository is the official implementation*

### Prerequisites
The code was developed and tested under Ubuntu 20.04 and python version 3.8.

Install all dependencies through

```
conda create --name fg python=3.8
conda activate fg
sudo apt-get install -y libsuitesparse-dev libboost-all-dev libcholmod3
pip install -r requirements.txt
```

*We also highly recommend, if possible, to use JAX with GPU/TPU-support. See the official [Documentation](https://github.com/google/jax#pip-installation-gpu-cuda) for install instructions*

### General Purpose Estimation
In `minimal_example.py` we provide a small (non-) working example on how to use the factor graph as a stand-alone method to integrate in your workflow.

### Just Part Poses
We provide a simple example on how to use the factor graph with pre-computed poses in the `only_poses`-directory. To execute the example run
```
python -m only_poses.main \
    --experiment-root-path ./experiments/only_poses \ 
    --experiment-name <my_experiment> \ 
    --motion-type <TRANS or ROT> \  # Which motion type we generate examples for
    --stddev-pos <Noise on pose position in meters> \
    --stddev-ori <Noise on pose orientation in radians> \ 
    --sample-length <Length of the sample/trajectory, for TRANS: in meters, for ROT in radians> \ 
    --observation-amount <Observations per sample/trajectory> \ 
    --number-samples <Number of samples created>
```

The script first checks if there are already pre-generated samples available (overwrite with `--create-samples` to always create new samples) and then runs the inference. Deactivate methods with
```
  --no-use-sturm
  --no-use-sturm-original
  --no-use-fg
  --no-use-fg-gt
```

An example could be
```
python -m only_poses.main \
    --experiment-root-path ./experiments/only_poses \
    --experiment-name testing \
    --motion-type TRANS \
    --stddev-pos 0.03 \
    --stddev-ori 0.01 \
    --sample-length 0.5 \
    --observation-amount 10 \
    --number-samples 10
```

For evaluation open the notebook `evaluation.ipynb` to analyze the results. In the first cell update
```
os.chdir("/root/of/the/repo/") 
```
to point to the root of the repository. The notebook automatically gathers all experiments in the given root-path.

Auto-generate commands for sample settings used in the paper through running
```
python -m only_poses.run_all
```
and prints them to the console.

### Workflow for Full RGB-D Experiments
Available upon request.

## 👩‍⚖️ License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.

## 🤝 Acknowledgements
Toyota Research Institute (TRI) provided funds to assist the authors with their research but this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.

---

If you find our work useful, please consider citing our paper:
```
@inproceedings{DBLP:conf/iros/HeppertMYCB22,
  author    = {Nick Heppert and
               Toki Migimatsu and
               Brent Yi and
               Claire Chen and
               Jeannette Bohg},
  title     = {Category-Independent Articulated Object Tracking with Factor Graphs},
  booktitle = {{IEEE/RSJ} International Conference on Intelligent Robots and Systems,
               {IROS} 2022, Kyoto, Japan, October 23-27, 2022},
  pages     = {3800--3807},
  publisher = {{IEEE}},
  year      = {2022},
  doi       = {10.1109/IROS47612.2022.9982029},
}
```
