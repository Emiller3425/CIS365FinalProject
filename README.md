# CIS365FinalProject
Trevor Martin, Ethan Miller, Tristan Ingram-Reeves, Evan Yu

## Writeup document with sources
https://docs.google.com/document/d/1gTMFagYb_E1eFhvKD_6BamGArHl-ue_H8tZAVVjIZxU/edit?usp=sharing

## Slides
https://docs.google.com/presentation/d/1T4yx4UwE5WgTHELnsCpXiRWDwnVGYCBCMg71BIoqo4Y/edit?usp=sharing

## Instructions
To run the BCBot.py or SmashTrainer.py files, which correlate with the Behavioral Cloning and Reinforcement Learning
approaches respectively, first install Slippi Online from https://slippi.gg/, and ensure that a version of Super Smash
Bros. Melee, NTSC, v1.2 is present in the same directory as the script. The scripts expect a file named "ssb.iso," but you
can change the filename if you'd like by changing the filename at line 32 of BCBot.py or line 162 of SmashGym.py. If you're
on Windows, these scripts will look for the Slippi Dolphin emulator in the default install directory, but if you're not, you'll
need to change the path on lines 152-153 of SmashGym.py or lines 12-13 of BCBot.py.

To run the MeleeBot.py script, which represents the basic decision tree agent, modify line 20 to point to your Slippi Dolphin
install, and modify line 38 to point to your image of Melee.

To run the BCNetDataGenerator.py script, change the paths on lines 112 and 113 to your desired input/output paths. input_path
is where your replays should be located, and output_path is where you'd like the generated arrays to be stored. Lines 122-123
were used to overcome a crash due to a corrupted replay file in our dataset, so unless you want to start at the 42017th replay,
remove those lines.

To run the BCNeuralNetwork.py script, change the load_directory value on line 16 to wherever your output files from
BCNetDataGenerator.py are.
