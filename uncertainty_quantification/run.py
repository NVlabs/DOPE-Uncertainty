import subprocess

# This is for test
obj = 'Corn'

# Step 1: inference
subprocess.call(['python', 'inference.py', \
	'--model', obj, obj+'_full', \
	'--object', obj, \
	'--data', 'output/test', \
	'--outf', 'output/test_result', \
	'--suffix', '*.png', \
	'--transform_visii', \
	'--config', '../config/config_pose_webcam_visii.yaml', \
	'--ensemble'])

# Step 2: post inference
subprocess.call(['python', 'post_inference.py', \
	'--model', obj, obj+'_full', \
	'--object', obj, \
	'--raw_data', 'output/test_result/raw_data.json', \
	'--resultf', 'output/test_result'])
