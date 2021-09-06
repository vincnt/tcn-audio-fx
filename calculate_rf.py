from utils import compute_receptive_field
import numpy as np 

combos = [(10,2,3,1),(6,7,20,1),(5,10,30,1), (4,25,20, 1), 
(6,8,20,1),(7,7,20,1),(10,2,100,1), (4, 10, 13, 1),
(4, 25, 20, 1),(4,10,300,1), (5,10,40,1),(8,4,20,1),(10,3,20,1)]

for i, combo in enumerate(combos):
    dilation_depth = combo[0]
    dilation_factor = combo[1]
    kernel_size = combo[2]
    num_repeat = combo[3]
    sr = 44100
    rf = compute_receptive_field(kernel_pattern=[kernel_size]*dilation_depth, 
                                dilation_pattern=[dilation_factor ** i for i in range(dilation_depth)]*num_repeat)
    rf_s =  np.round(rf/sr, 3)
    print(f"Combo{i} --  {combo} -- Receptive field: {rf_s} s")