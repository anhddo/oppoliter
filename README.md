# Optimistic policy iteration
## Algorithm
![Imgur](https://i.imgur.com/cfkk97w.png)
## Run
python tabular.py --n-episode 1000 --n-action 2 --n-step 3 --state-per-stage 2 --n-run 3
## Cumulative regret plot
Compare cumulative regret plot to **"Is Q-learning provably efficient?"** *(Chi Jin, et al 2018)*
![](https://i.imgur.com/BiwWceS.png)

## Tabular setting
python tabular.py --n-episode 100000 --n-action 100 --n-step 2 --state-per-stage 100 --c 0.1 --n-run 1

# Linear model.
## Optimistic value iteration for average reward problem.
![Imgur](https://i.imgur.com/5W5s77C.png)
Test on Open AI gym  
**Ex:** python leastsquare.py --fourier-order 4 --discount 0.99 --step 15000 --repeat 20 --bonus 1 --env MountainCar-v0 --lambda 1 --sample-len 5 --n-eval 1 --algo val --beta 20
