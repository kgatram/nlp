To run the network for inference, 
        1. Navigate into the submitted folder directory.
        2. Download the ‘.pt’ files of the networks from https://universityofstandrews907-my.sharepoint.com/:f:/g/personal/akhkr1_st-andrews_ac_uk/Eh0rOhTeM09Cmw3SIr13z2wBg8Vjpk2lDX7HbvJwuQylsA?e=8NI66j and save them in the submitted folder.
        3. Install the dependencies by running the command "pip install -r requirements.txt" and launch Jupyter notebook in the "submission" folder.
        4. Open the python notebook named ‘Inference.ipynb’.
        5. Enter the path of ".pt" file in 'model_name' as a string either ‘224_model’ or ‘32_model’.
        6. Enter the op_size (output size should match the model chosen) as 224 (for 224_model) or 32 (for 32_model) (integer).
        7. Enter the input_image_path as a string with the path to the input RGB image file.
        8. If the inference is only with input image then the next variable target_image_path can be left blank and run cells 1, 2, 3.
        9. If the target image is available for inference, enter the target_image_path as a string with the path to target depth image and run cells 1, 2, 4.
        10. The inference displays predicted pixel values, depth converted values for each pixel, plots the predicted and input images, prints loss if target is also given and plots the ground-truth images as well.

Project Structure:

Training code for 224_model					├── 224_model.py
								│
Training code for 32 model					├── 32_model.py
								│
Training code for network without DINO				├── without_dino.py
								│
Instructions to run inference					├── README.md
								│
Serialised scaler object					├── scaler.pkl
								│
Sample test image for inference					├── test_input.jpg
								│
Sample test target for inference				├── test_target.png
								│
Sample camera image for inference				├── chair.jpeg
								│
Dependencies list to run inference				├── requirements.txt
								│
Interactive python notebook for running inference		├── inference.ipynb
								│
Folder contains code of experimentations			├── experimental_networks
	Code for 4 layers network				│		 ├── 4_layers_network.py
								│		 │
	Code for feeding grayscale image to DINO		│		 ├── grayscale_to_dino.py
								│		 │
	Code for [64,64] grayscale target			│		 ├── gs_64.py
								│		 │
	Code that feeds camera input to DINO network		│		 ├── interface_dino.py
								│		 │
	Code for 5_5_5 network					│		 ├── original_size-2.py
								│		 │
	Code for generating downsized depth estimates		│		 ├── pred_dataset.py
								│		 │
	Code for super resolution model				│		 ├── super_res.py
								│		 │
	Code for training super resolution model		│		 └── train_super_res.py
								│
Folder contains miscellaneous files for reference		└── misc
									│
	Code used for Histogram generation				├── histogram.ipynb
									│
	Code used for latency evaluation				├── latency_check.ipynb
									│
	Code used for normalisation					├── normalisation.ipynb
									│
	Folder contains inference drawn from 224 model with test data	├── 224_test
								 	│	├── 224_network_test.txt
									│	├── input_0.jpg
								  	│	├── input_80.jpg
								 	│	├── input_90.jpg
									│	├── pred_0.png
								 	│	├── pred_80.png
									│	├── pred_90.png
								 	│	├── target_0.png
									│	├── target_80.png
									│	└── target_90.png
									│
	Code for bayesian optimisation using optuna			├── bayes_optim_depth.py
									│
	Code for computing features dataset using SIFT			├── compute_pyramid_and_sift.ipynb
	Draft for the same						├── compute_pyramid_and_sift_draft.ipynb
									│
	Pdf which shows the data flow					├── data_flow.pdf
									│
	DINO outputs for experimenting with it				├── dino_output.txt
									│
	Plotting the attention applied by DINO				├── dinov2_with_image_experiment.ipynb
									│
	Experimenting with NYU labeled dataset				├── labeled_dataset.ipynb
									│
	Code for syncing the RGB images with corresponding targets	├── sync.ipynb
									│
	Code for merging the tum dataset				└── tum_merge.ipynb
