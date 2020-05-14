from dm_control import suite
import numpy as np
# Load one task :
env = suite.load ( domain_name ="cartpole", task_name ="swingup" )
spec = env.action_spec ()
time_step = env.reset ()
while not time_step.last():
    action = np.random.uniform( spec.minimum , spec.maximum , spec.shape  )
    print(action)
    time_step = env.step ( action  )
    print(time_step)
