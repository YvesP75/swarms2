# swarms2

**red swarms against blue swarms**
reds attack a target defended by blues
reds and blues may kill their opponent
the more drones reach the goal, the better for the reds and the worst for the blues

please install the requirements in requirement.txt 
code has been tested on MAC OS with Python3.9
Pillow may have some difficulties in its installation. A possibility is to try falling back to Python3.7

All files are flat.
two files contain the starting points :
train.py allows for training
show.py launches the streamlit app

**train.py**
the training is done iteratively by starting with one blue drone and one red drone, increasing distance progressively and adding drones.
Learned policies are stored in /policies/
For each configuration depending on the number of blue and red drones, there is a folder containing the red and blue policies
ex /policies/b3r4 contains the red and blue policies for configurations with 3 blue and 4 red drones 
There are two policies depending on the color : one for blues and one for reds. Several intermediate savings are done. the last one being __last.
The function to launch for the training is : 
super_meta_train(max_blues=8, max_reds=8, iteration=10, max_dispersion=3, total_timesteps=10000)
which programs a training from 1,1 to 8,8 drones, with a distance mutltiplier of 3 and 10 iterations, and a total timestep at each learning of 10000 steps
(1 step = 1 second)


**show.py**
once the agents are trained, the drones can be simulated and visible through a streamlit interface.
the command to launch the visualisation is :
streamlit run show.py


**tuning the rewards**
The rewards may be tuned in the _param.py file and logic is in the team_wrap.py file in the 'evaluate_situation' function.
When is_double is true, means that there is no learning: simulation is carried out with already defined policies. Only the final outcome is to be considered.
Otherwise, two cases have to be taken into account, whether blue or red is learning