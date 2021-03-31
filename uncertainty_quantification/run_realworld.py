import subprocess

# This is for test (the real world case, w/o ground truth)
obj = 'Corn'

# Step 1: inference
subprocess.call(['python', 'inference.py', \
	'--model', obj, obj+'_full', \
	'--object', obj, \
	'--data', 'output/test', \
	'--outf', 'output/test_result_realworld', \
	'--suffix', '*.png', \
	'--transform_visii', \
	'--config', '../config/config_pose_webcam_visii.yaml', \
	'--ensemble', '--real_world'])

# Step 2: post inference
subprocess.call(['python', 'post_inference.py', \
	'--model', obj, obj+'_full', \
	'--object', obj, \
	'--raw_data', 'output/test_result_realworld/raw_data.json', \
	'--resultf', 'output/test_result_realworld', \
	'--real_world'])