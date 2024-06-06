face_embeddings.npy
	-contains face embedding data of 2.6Lac samples. shape(num_samples,1,512)

build_cluster.py
	-run this python script, which uses face_embeddings.npy to create two files,
		1) cluster_tree.csv - face embedding data sorted according to the cluster numbers. shape(num_samples,516).
			Four columns corresponding to layer numbers have been appended and data sorted according to them.
		2) output_dict.json - output dictionary with key:value pairs, corresponding to cluster_number:centroid

detect_face.py
	-module to detect and align face from raw images. Useful if testing for a new image which needs face detection and alignment.

final_run_model.py
	-module to test clustering model. Contains methods to return list of cluster numbers shape(4,) and scores corresponding to each
	data in final landing cluster.

aligned_test_images
	-contains aligned images to test model.
		1)Images with name like "{}_0" are the images model was trained on. So it should always
	land in the right cluster, with score 0.99
		2)Other images not of the above format are other images of the people model is trained on. This doesn't always land in
	the right cluster. If it lands in right cluster, gives good face recognition score values around 0.7-0.8. Probabilty of correct
	cluster prediction is more if the test and training image both perfectly show frontal face.


