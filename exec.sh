for((i=0.31;i<=0.32;i+=0.01));  
do   
python cluster_test.py --method DBSCAN --videoDir /workspace/data/blue/feature/data_blue_feature --picDir /workspace/data/blue/crop_faces --saveResult True --saveDir result --eps $i;
done 