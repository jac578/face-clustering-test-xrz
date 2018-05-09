python cluster_test.py --method DBSCAN --videoDir /workspace/data/lfw-features/insightface-r100-spa-m2.0-ep96 --featureList /workspace/data/lfw-features/nosingle_lfw_align_succeeded_feature_list.txt \
--picDir /workspace/data/lfw/LFW-mtcnn-simaligned-112x112 \
--saveResult True \
--saveDir ./nosingle_clustering_result \
--eps 0.45 --nProcess 16 --evaluate True \
--labelDict /workspace/data/lfw-features/insightface-r100-spa-m2.0-ep96