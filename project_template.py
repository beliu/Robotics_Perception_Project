#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

	# TODO: Convert ROS msg to PCL data
	cloud = ros_to_pcl(pcl_msg)

	# TODO: Statistical Outlier Filtering
	cloud_filter = cloud.make_statistical_outlier_filter()
	# Set the number of neighboring points to analyze for any given point
	cloud_filter.set_mean_k(50)
	# Set threshold scale factor
	x = 0.2
	# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
	cloud_filter.set_std_dev_mul_thresh(x)
	# Finally call the filter function
	cloud_filtered = cloud_filter.filter()

	# TODO: Voxel Grid Downsampling
	# Create a VoxelGrid filter object for our input point cloud
	vox = cloud_filtered.make_voxel_grid_filter()
	# Choose a voxel (also known as leaf) size
	LEAF_SIZE = 0.01

	# Set the voxel (or leaf) size
	vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

	# Call the filter function to obtain the resultant downsampled point cloud
	cloud_filtered = vox.filter()

	# TODO: PassThrough Filter
	# Create a PassThrough filter object.
	passthrough = cloud_filtered.make_passthrough_filter()

	# Assign axis and range to the passthrough filter object.
	filter_axis = 'z'
	passthrough.set_filter_field_name(filter_axis)
	axis_min = 0.6
	axis_max = 1.1
	passthrough.set_filter_limits(axis_min, axis_max)

	# Finally use the filter function to obtain the resultant point cloud. 
	cloud_filtered = passthrough.filter()

	# Create a PassThrough filter object.
	passthrough = cloud_filtered.make_passthrough_filter()

	# Assign axis and range to the passthrough filter object.
	filter_axis = 'y'
	passthrough.set_filter_field_name(filter_axis)
	axis_min = -0.4
	axis_max = 0.4
	passthrough.set_filter_limits(axis_min, axis_max)

	# Finally use the filter function to obtain the resultant point cloud. 
	cloud_filtered = passthrough.filter()

	# TODO: RANSAC Plane Segmentation
	# Create the segmentation object
	seg = cloud_filtered.make_segmenter()

	# Set the model you wish to fit 
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)

	# Max distance for a point to be considered fitting the model
	# Experiment with different values for max_distance 
	# for segmenting the table
	max_distance = 0.01
	seg.set_distance_threshold(max_distance)

	# Call the segment function to obtain set of inlier indices and model coefficients
	inliers, coefficients = seg.segment()

	# TODO: Extract inliers and outliers
	# Extract inliers
	cloud_table = cloud_filtered.extract(inliers, negative=False)
	# Extract outliers
	cloud_objects = cloud_filtered.extract(inliers, negative=True)

	# TODO: Euclidean Clustering
	white_cloud = XYZRGB_to_XYZ(cloud_objects)# Apply function to convert XYZRGB to XYZ
	tree = white_cloud.make_kdtree()

	# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
	# Create a cluster extraction object
	ec = white_cloud.make_EuclideanClusterExtraction()

	# Set tolerances for distance threshold 
	# as well as minimum and maximum cluster size (in points)
	ec.set_ClusterTolerance(0.02)
	ec.set_MinClusterSize(50)
	ec.set_MaxClusterSize(2000)

	# Search the k-d tree for clusters
	ec.set_SearchMethod(tree)

	# Extract indices for each of the discovered clusters
	cluster_indices = ec.Extract()

	# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
	#Assign a color corresponding to each segmented object in scene
	cluster_color = get_color_list(len(cluster_indices))

	color_cluster_point_list = []

	for j, indices in enumerate(cluster_indices):
	    for i, indice in enumerate(indices):
	        color_cluster_point_list.append([white_cloud[indice][0],
	                                         white_cloud[indice][1],
	                                         white_cloud[indice][2],
	                                         rgb_to_float(cluster_color[j])])

	#Create new cloud containing all clusters, each with unique color
	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)

	# TODO: Convert PCL data to ROS messages
	ros_cloud_objects =  pcl_to_ros(cloud_objects)
	ros_cloud_table = pcl_to_ros(cloud_table)
	ros_cluster_cloud = pcl_to_ros(cluster_cloud)

	# TODO: Publish ROS messages
	pcl_objects_pub.publish(ros_cloud_objects)
	pcl_table_pub.publish(ros_cloud_table)
	pcl_cluster_pub.publish(ros_cluster_cloud)




	# Classify the clusters! (loop through each detected cluster one at a time)
	detected_objects_labels = []
	detected_objects = []

	for index, pts_list in enumerate(cluster_indices):
		
		# Grab the points for the cluster
		pcl_cluster = cloud_objects.extract(pts_list)
		ros_cluster = pcl_to_ros(pcl_cluster)

		# Compute the associated feature vector
		chists = compute_color_histograms(ros_cluster, using_hsv=True)
		normals = get_normals(ros_cluster)
		nhists = compute_normal_histograms(normals)
		feature = np.concatenate((chists, nhists))

		# Make the prediction
		prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
		prob = clf.predict_proba(scaler.transform(feature.reshape(1,-1)))
		label = encoder.inverse_transform(prediction)[0]
		max_prob = round(np.max(prob), 2)
		

		if max_prob > 0.5:
			detected_objects_labels.append(label)

			# Publish a label into RViz
			label_pos = list(white_cloud[pts_list[0]])
			label_pos[2] += .4
			object_markers_pub.publish(make_label(label + ': ' + str(max_prob),label_pos, index))

			# Add the detected object to the list of detected objects.
			do = DetectedObject()
			do.label = label
			do.cloud = ros_cluster
			detected_objects.append(do)

	# Publish the list of detected objects
	rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

	# Publish the list of detected objects
	detected_objects_pub.publish(detected_objects)


	# Suggested location for where to invoke your pr2_mover() function within pcl_callback()
	# Could add some logic to determine whether or not your object detections are robust
	# before calling pr2_mover()
	try:
		pr2_mover(detected_objects)
	except rospy.ROSInterruptException:
	    pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

	# TODO: Initialize variables
	test_scene_num = Int32()
	test_scene_num.data = 1
	object_name = String()
	arm_name = String()
	pick_pose = Pose()
	place_pose = Pose()
	label_list = []
	centroid_list = []
	dict_list = []

	# TODO: Get/Read parameters
	object_list_param = rospy.get_param('/object_list')
	place_pose_param = rospy.get_param('/dropbox')

	# TODO: Get the PointCloud for a given object and obtain it's centroid
	for object_ in object_list:
		label_list.append(object_.label)
		points_arr = ros_to_pcl(object_.cloud).to_array()
		centroid = np.mean(points_arr, axis=0)[:3]
		centroid = [np.asscalar(p) for p in centroid]
		centroid_list.append(centroid)

	# TODO: Loop through the pick list
	for i in range(len(object_list_param)):
		object_name.data = object_list_param[i]['name']

		try:
			idx = np.where(np.array(label_list) == object_list_param[i]['name'])[0]
			centroid = centroid_list[idx[0]]
			pick_pose.position.x = centroid[0]
			pick_pose.position.y = centroid[1]
			pick_pose.position.z = centroid[2]
			if (object_list_param[i]['group'] == 'green'):
				# TODO: Assign the arm to be used for pick_place
				arm_name.data = 'right'
				# TODO: Create 'place_pose' for the object
				arm = 1	    	
			elif (object_list_param[i]['group'] == 'red'):
				arm_name.data = 'left'
				arm = 0
			place_pose.position.x = place_pose_param[arm]['position'][0]
			place_pose.position.y = place_pose_param[arm]['position'][1]
			place_pose.position.z = place_pose_param[arm]['position'][2]
			# TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
			yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
			dict_list.append(yaml_dict)
		except:
			print(object_name.data, ' was not detected')

	# TODO: Output your request parameters into output yaml file
	send_to_yaml('output_1.yaml', dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    # Detection and clustering publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Labels and Detected Objects publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
 		  rospy.spin()