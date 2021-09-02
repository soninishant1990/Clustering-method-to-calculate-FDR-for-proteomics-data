import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import revoscalepy as revoscale
from scipy.spatial import distance as sci_distance
from sklearn import cluster as sk_cluster
from sklearn.preprocessing import StandardScaler
import csv
import re
import os
import sys

#example = python .\clustering_kmean-3-3_4.py Target_file_for_cluster.csv Pass_PSM_file.csv
script = sys.argv[0]
filename = sys.argv[1]
data = pd.read_csv(filename)
print(data.head())


#	Determine number of clusters using the Elbow method

#cdata = data
#K = range(1, 20)
#KM = (sk_cluster.KMeans(n_clusters=k).fit(cdata) for k in K)
#centroids = (k.cluster_centers_ for k in KM)

#D_k = (sci_distance.cdist(cdata, cent, 'euclidean') for cent in centroids)
#dist = (np.min(D, axis=1) for D in D_k)
#avgWithinSS = [sum(d) / cdata.shape[0] for d in dist]
#plt.plot(K, avgWithinSS, 'b*-')
#plt.grid(True)
#plt.xlabel('Number of clusters')
#plt.ylabel('Average within-cluster sum of squares')
#plt.title('Elbow for KMeans clustering')
#plt.show()

################################################################################################
##	Perform clustering using Kmeans
################################################################################################

# It looks like k=4 is a good number to use based on the elbow graph.
n_clusters = 2

means_cluster = sk_cluster.KMeans(n_clusters=n_clusters, random_state=111)
#columns = ['experimentalMassToCharge', 'calculatedMassToCharge', 'DM',
       #'absdM', 'total_b_ion_match', 'total_y_ion_match',
      # 'Total_b_y_ion_match', 'Total_b_y_fraction', 'IsotopeError',
      # 'charge_state', 'precursorError', 'miss_cleavage',
       #'Energy', 'enzN', 'enzC', 'MSGF_RawScore', 'MSGF_DeNovoScore', 
	   #'MSGF_SpecEValue', 'MSGF_EValue', 
	   #'ExplainedIonCurrentRatio',
	  # 'NTermIonCurrentRatio', 'CTermIonCurrentRatio', 'MS2IonCurrent', 'NumMatchedMainIons',
      # 'ScoreRatio']


#columns = ['DM', 'absdM', 'total_b_ion_match', 'total_y_ion_match',
       #'Total_b_y_ion_match', 'Total_b_y_fraction','IsotopeError',
       #'precursorError', 'Tryptic', 'absPrecursorError',
       #'Energy','MSGF_RawScore', 'MSGF_DeNovoScore', 'peptide_legth_true_false', 
	   #'MSGF_SpecEValue', 'MSGF_EValue', 'ExplainedIonCurrentRatio',
	   #'NTermIonCurrentRatio', 'CTermIonCurrentRatio', 'MS2IonCurrent', 'NumMatchedMainIons',
       #'ScoreRatio','absEnergy','absMSGF_RawScore','absMSGF_DeNovoScore','absScoreRatio',] #'q-value','pep_q-value'

#columns = ['DM', 'absdM', 'total_b_ion_match', 'total_y_ion_match',
       #'Total_b_y_ion_match', 'Total_b_y_fraction','IsotopeError',
       #'precursorError', 'Tryptic','absPrecursorError',
       #'Energy','MSGF_RawScore', 'MSGF_DeNovoScore', 'peptide_legth_true_false', 
	   #'MSGF_SpecEValue', 'MSGF_EValue', 'ExplainedIonCurrentRatio',
	   #'NTermIonCurrentRatio', 'CTermIonCurrentRatio', 'MS2IonCurrent', 'NumMatchedMainIons',
       #'ScoreRatio','absEnergy','absMSGF_RawScore','absMSGF_DeNovoScore','absScoreRatio',] #'q-value','pep_q-value'

columns = ['absMSGF_RawScore','MSGF_RawScore','absPrecursorError','IsotopeError','DM','Energyevalue','MSGF_EValue','absIsotopeError','MS2IonCurrent'] #'q-value','pep_q-value'
#columns = [ 'absdM','absPrecursorError','absIsotopeError','nag_absMSGF_RawScore','MSGF_EValue','absEnergyevalue'] #'q-value','pep_q-value'



print(columns)
select_df = data[columns]
select_df1 = np.nan_to_num(select_df)


#Scale the Features using StandardScaler
X = StandardScaler().fit_transform(select_df1)

model = means_cluster.fit(X)
clusters = model.labels_
data['cluster_result'] = clusters
print(data)

# Print some data about the clusters:

# For each cluster, count the members.
d =0
for c in range(n_clusters):
	cluster_members=data[data['cluster_result'] == c][:]
	print('cluster_result{}(n={}):'.format(c, len(cluster_members)))
	print('-'* 17)
	if d == 0:
		cluster0=len(cluster_members)
	else:
		cluster1=len(cluster_members)
	d +=1

# Print mean values per cluster.
print(data.groupby(['cluster_result']).mean())

#plt.scatter(X[:,33],X[:,15],c=model.labels_, cmap='rainbow')
#plt.scatter(means_cluster.cluster_centers_[:,33] ,means_cluster.cluster_centers_[:,15], color='black')
#plt.show()

print('cluster0=',cluster0)
print('cluster1',cluster1)

cluster = []
if cluster0 < cluster1:
	for row in data['cluster_result']:
		if row == 0:
			cluster.append('cluster0')
		else:
			cluster.append('cluster1')
else:
	for row in data['cluster_result']:
		if row == 0:
			cluster.append('cluster1')
		else:
			cluster.append('cluster0')

data['cluster'] = cluster

data.rename(columns={'ID':'spectrumID'},index={0:'zero',1:'one'},inplace=True)

data.to_csv('result.csv', index=False)





#!/usr/bin/python
#date 20-june-2019
# code for weka generated 2 cluster file to analysis of accuracy and sensitivity [project -  identify correct and incorrect PSM in mass spectrometry data though machine learning]
import csv
import re
import os
import sys
import shutil
import pandas as pd 
import pandas
import matplotlib.pyplot as plt
import numpy as np
import math


files1 = 'result.csv'
filename1 = sys.argv[2]
#mainefile_with_all_parameter = 'LTQFT_1562653383_text_Separate_FDR.csv'
mainefile_with_all_parameter = filename1
Resultfile = open(files1+"result.txt","w")


files = files1
#convert arff file to csv file
with open(files,'r', newline='') as f, open(files,'r', newline='') as fileheader, open(files+'_ result_cluster0.csv','w', newline='') as cluster0:
	csv_reader = csv.reader(f, delimiter=',')
	store_values1 = csv.writer(cluster0, delimiter=',')
	next(csv_reader)
	for headingline1 in fileheader:
		headingline2 = headingline1.split(',')
		store_values1.writerow(headingline2)
		#print(headingline2)
		break;
	#store_values1 = csv.writer(cluster0, delimiter=',')
	store_values = cluster0
	for num in csv_reader:
		try:
			if "cluster0" ==num[-1]:
				store_values1.writerow(num)
		except IndexError:
			pass
		



with open(files,'r', newline='') as f, open(files,'r', newline='') as fileheader1, open(files+'_result_cluster1.csv','w',newline='') as cluster1:
	csv_reader = csv.reader(f, delimiter=',')
	next(csv_reader)
	store_values4 = csv.writer(cluster1, delimiter=',')
	store_values3 = cluster1
	for headingline1 in fileheader1:
		headingline2 = headingline1.split(',')
		store_values4.writerow(headingline2)
		#print(headingline2)
		break;
	for num in csv_reader:
		try:
			if "cluster1" ==num[-1]:
				store_values4.writerow(num)
		except IndexError:
			pass

#apply 1% FDR PSM file
PSM_file = pd.read_csv(mainefile_with_all_parameter)
q_value = PSM_file[PSM_file['q-value'] <=0.01]
q_value.to_csv(mainefile_with_all_parameter+'1%.csv')
#print(q_value)
spectrafidfile = mainefile_with_all_parameter+'1%.csv'
#sys.exit()


# Copy unique id from FDR file to new csv file for comparision
cols = ['number']
fdrspectrafile = spectrafidfile
df = pd.read_csv(fdrspectrafile, usecols=cols)
df = df.astype(int)

df.to_csv("spectrumid"+fdrspectrafile, index=False)




# files name
clusterfile0 = files+'_ result_cluster0.csv'
clusterfile1 = files+'_result_cluster1.csv'
spectraidfile = "spectrumid"+fdrspectrafile


#copy spectra unique number in csv file for comparison
cols = ['spectrumID']
spectrafile = clusterfile0
spectrafile1 = clusterfile1
pd.read_csv(spectrafile, usecols=cols).to_csv(clusterfile0+"_onlyspectra_result.csv", index=False)
pd.read_csv(spectrafile1, usecols=cols).to_csv(clusterfile1+"_onlyspectra_result.csv", index=False)

#Count lines in csv file
resultfile = pd.read_csv(clusterfile0+"_onlyspectra_result.csv")
lines = resultfile.count()
lines3 = str(lines )
print('cluster0_Resultfile_PSM = ',lines)
Resultfile.write('cluster0_Resultfile_PSM = ')
Resultfile.write(lines3)
Resultfile.write('\n')

resultfile1 = pd.read_csv(clusterfile1+"_onlyspectra_result.csv")
lines1 = resultfile1.count()
lines4 = str(lines1)
print('cluster1_Resultfile_PSM = ',lines1)
Resultfile.write('cluster1_Resultfile_PSM = ')
Resultfile.write(lines4)
Resultfile.write('\n')


spectrafile = pd.read_csv(spectraidfile)
lines2 = spectrafile.count()
lines5 = str(lines2)
print('FDR_file_PSM = ',lines2)
Resultfile.write('FDR_file_PSM = ',)
Resultfile.write(lines5)
Resultfile.write('\n')
Resultfile.write('---------------------------------------------------------------------------------------------')
Resultfile.write('\n')

#set_diff_df = pd.concat([df2, df1, df1]).drop_duplicates(keep=False)



#Match FDR spectra resullt with cluster result
with open(clusterfile0+"_onlyspectra_result.csv",'r', newline='') as f, open(spectraidfile,'r', newline='') as fh, open(clusterfile0+'_cluster0_match_with_fdr_result.csv','w',newline='') as cluster_result, open(clusterfile0+'_not_match_with_fdr_result_result.csv','w',newline='') as fdr_result:
	csv_reader = csv.reader(f, delimiter=',')
	csv_reader1 = csv.reader(fh, delimiter=',')
	cluster_result1 = csv.writer(cluster_result, delimiter=',')
	fdr_result1 = csv.writer(fdr_result, delimiter=',')
	x = []
	for n in csv_reader1:
		e = n
		for lines in e:
			x.append(lines)
	#print("x",x)
	
	for m in csv_reader:
		d = m
		for line in d:
			if line in x:
				#print("line",line)
				#print("x",x)
				#print("match")
				cluster_result.write(line)
				cluster_result.write('\n')
			else:
				#print("not_match")
				fdr_result.write(line)
				fdr_result.write('\n')


with open(clusterfile1+"_onlyspectra_result.csv",'r', newline='') as f1, open(spectraidfile,'r', newline='') as fh1, open(clusterfile1+'_cluster1_match_with_fdr_result.csv','w',newline='') as cluster_result1, open(clusterfile1+'_not_match_with_fdr_result_result.csv','w',newline='') as fdr_result1:
	csv_reader1 = csv.reader(f1, delimiter=',')
	csv_reader11 = csv.reader(fh1, delimiter=',')
	cluster_result0 = csv.writer(cluster_result1, delimiter=',')
	fdr_result0 = csv.writer(fdr_result1, delimiter=',')
	x1 = []
	for n1 in csv_reader11:
		e1 = n1
		for lines1 in e1:
			x1.append(lines1)
	#print("x",x)
	for m1 in csv_reader1:
		d1 = m1
		for line1 in d1:
			if line1 in x1:
				#print("line",line)
				#print("x",x)
				#print("match")
				cluster_result1.write(line1)
				cluster_result1.write('\n')
			else:
				#print("not_match")
				fdr_result1.write(line1)
				fdr_result1.write('\n')
	



#generate new file of match result with all parameter
cluster0_match_file = clusterfile0+'_cluster0_match_with_fdr_result.csv'
cluster0_notmatch_file = clusterfile0+'_not_match_with_fdr_result_result.csv'
cluster1_match_file = clusterfile1+'_cluster1_match_with_fdr_result.csv'
cluster1_notmatch_file = clusterfile1+'_not_match_with_fdr_result_result.csv'

mainefile7 = mainefile_with_all_parameter
mainefiles8 = mainefile_with_all_parameter
mainefile1 = mainefile_with_all_parameter
mainefiles2 = mainefile_with_all_parameter
mainefile3 = mainefile_with_all_parameter
mainefiles4 = mainefile_with_all_parameter
mainefile5 = mainefile_with_all_parameter
mainefiles6 = mainefile_with_all_parameter



with open(cluster0_match_file,'r', newline='') as cluster0_match_file, open(mainefile7,'r', newline='') as mainefile, open(mainefiles8,'r', newline='') as headingline, open('cluster0_match_with_fdr_result_all_paremeter.csv','w',newline='') as cluster_result_with_all_perameter:
	cluster_result_with_all_perameter1 = csv.writer(cluster_result_with_all_perameter, delimiter=',')

	for headingline1 in headingline:
		headingline2 = headingline1.split(',')
		cluster_result_with_all_perameter1.writerow(headingline2)
		#print(headingline2)
		break;
	
	x = 0
	d = dict()
	line =mainefile.readline()
		
	for line in mainefile:
		line1 = line.split(',')
		key = line1[0]
		key = int(key)
		d[key] = ''
		val = line1[0:]
		d[key] = val
		x+=1
	#print(d)


	for match in cluster0_match_file:
		match = match.rstrip("\n")
		match = int(match)
		#print(match)
		findresult = d[match]
		#print(findresult)
		cluster_result_with_all_perameter1.writerow(findresult)

	cluster_result_with_all_perameter.close()
	
	

with open(cluster0_notmatch_file,'r', newline='') as cluster0_notmatch_file, open(mainefile1,'r', newline='') as mainefile1, open(mainefiles2,'r', newline='') as headingline5, open('cluster0_not_match_with_fdr_result_all_paremeter.csv','w',newline='') as cluster_result_with_all_perameter2:
	cluster_result_with_all_perameter7 = csv.writer(cluster_result_with_all_perameter2, delimiter=',')

	for headingline1 in headingline5:
		headingline2 = headingline1.split(',')
		cluster_result_with_all_perameter7.writerow(headingline2)
		#print(headingline2)
		break;
	
	x = 0
	d = dict()
	line =mainefile1.readline()
		
	for line in mainefile1:
		line1 = line.split(',')
		key = line1[0]
		key = int(key)
		d[key] = ''
		val = line1[0:]
		d[key] = val
		x+=1
	#print(d)


	line10 =cluster0_notmatch_file.readline()

	try:
		for match in cluster0_notmatch_file:
			match = match.rstrip("\n")
			match = int(match)
			#print(match)
			findresult = d[match]
			#print(findresult)
			cluster_result_with_all_perameter7.writerow(findresult)
	except:
		pass
	
	#cluster_result_with_all_perameter2.close()



	
with open(cluster1_match_file,'r', newline='') as cluster1_match_file, open(mainefile3,'r', newline='') as mainefile3, open(mainefiles4,'r', newline='') as headingline10, open('cluster1_match_with_fdr_result_all_paremeter.csv','w',newline='') as cluster_result_with_all_perameter3:
	cluster_result_with_all_perameter4 = csv.writer(cluster_result_with_all_perameter3, delimiter=',')

	for headingline9 in headingline10:
		headingline8 = headingline9.split(',')
		cluster_result_with_all_perameter4.writerow(headingline8)
		#print(headingline2)
		break;
	
	x = 0
	d = dict()
	line =mainefile3.readline()
		
	for line in mainefile3:
		line1 = line.split(',')
		key = line1[0]
		key = int(key)
		d[key] = ''
		val = line1[0:]
		d[key] = val
		x+=1
	#print(d)

	for match in cluster1_match_file:
		match = match.rstrip("\n")
		match = int(match)
		#print(match)
		findresult = d[match]
		#print(findresult)
		cluster_result_with_all_perameter4.writerow(findresult)



with open(cluster1_notmatch_file,'r', newline='') as cluster1_notmatch_file, open(mainefile5,'r', newline='') as mainefile, open(mainefiles6,'r', newline='') as headingline, open('cluster1_not_match_with_fdr_result_all_paremeter.csv','w',newline='') as cluster_result_with_all_perameter5:
	cluster_result_with_all_perameter6 = csv.writer(cluster_result_with_all_perameter5, delimiter=',')

	for headingline1 in headingline:
		headingline2 = headingline1.split(',')
		cluster_result_with_all_perameter6.writerow(headingline2)
		#print(headingline2)
		break;
	
	x = 0
	d = dict()
	line =mainefile.readline()
		
	for line in mainefile:
		line1 = line.split(',')
		key = line1[0]
		key = int(key)
		d[key] = ''
		val = line1[0:]
		d[key] = val
		x+=1
	#print(d)

	line11 =cluster1_notmatch_file.readline()

	
	try:
		for match in cluster1_notmatch_file:
			match = match.rstrip("\n")
			match = int(match)
			#print(match)
			findresult = d[match]
			#print(findresult)
			cluster_result_with_all_perameter6.writerow(findresult)
	except KeyError:  pass
	


#calculate accuracy and sencitivity(True positive rate)
cluster0_match_files = clusterfile0+'_cluster0_match_with_fdr_result.csv'
cluster0_notmatch_files = clusterfile0+'_not_match_with_fdr_result_result.csv'
cluster1_match_files = clusterfile1+'_cluster1_match_with_fdr_result.csv'
cluster1_notmatch_files = clusterfile1+'_not_match_with_fdr_result_result.csv'


cluster0_match_file1 = pd.read_csv(cluster0_match_files)
cluster0_match_file_lines = cluster0_match_file1.count()
cluster0_match_file_lines1 = cluster0_match_file_lines+1
cluster0_match_file_lines+=1
cluster0_match_file_lines2 = str(cluster0_match_file_lines1)
print('cluster0_match_with_fdr_result (TP) = ',cluster0_match_file_lines1)
Resultfile.write('cluster0_match_with_fdr_result = ')
Resultfile.write(cluster0_match_file_lines2)
Resultfile.write('\n')

cluster0_notmatch_file1 = pd.read_csv(cluster0_notmatch_files)
cluster0_notmatch_file_lines = cluster0_notmatch_file1.count()
cluster0_notmatch_file_lines1 = str(cluster0_notmatch_file_lines)
print('cluster0_not_match_with_fdr_result (FP) = ',cluster0_notmatch_file_lines)
Resultfile.write('cluster1_not_match_with_fdr_result = ')
Resultfile.write(cluster0_notmatch_file_lines1)
Resultfile.write('\n')

col_Names=["spectrum"]
cluster1_match_file1 = pd.read_csv(cluster1_match_files, names=col_Names)
cluster1_match_file_lines = cluster1_match_file1.count()
cluster1_match_file_lines1 = cluster1_match_file_lines
cluster1_match_file_lines2 = str(cluster1_match_file_lines1)
print('cluster1_match_with_fdr_result (FN) = ',cluster1_match_file_lines1)
Resultfile.write('cluster1_match_with_fdr_result = ')
Resultfile.write(cluster1_match_file_lines2)
Resultfile.write('\n')

cluster1_notmatch_file1 = pd.read_csv(cluster1_notmatch_files)
cluster1_notmatch_file_lines = cluster1_notmatch_file1.count()
cluster1_notmatch_file_lines1 = str(cluster1_notmatch_file_lines)
print('cluster1_not_match_with_fdr_result (TN) = ',cluster1_notmatch_file_lines)
Resultfile.write('cluster1_not_match_with_fdr_result = ')
Resultfile.write(cluster1_notmatch_file_lines1)
Resultfile.write('\n')
Resultfile.write('---------------------------------------------------------------------------------------------')
Resultfile.write('\n')


#calculate accuracy and sensitivity and MCC
TP1 =cluster0_match_file_lines
FP2=cluster0_notmatch_file_lines
TN3=cluster1_notmatch_file_lines
FN4 = cluster1_match_file_lines


TP = int(TP1)
FP = int(FP2)
TN = int(TN3)
FN = int(FN4)

accuracy1= TP+TN
accuracy2= TP+TN+FN+FP
accuracy3= accuracy1/accuracy2
accuracy4 = accuracy3 *100
print('accuracy =  ',accuracy4)
accuracy5 = str(accuracy4)
Resultfile.write('accuracy =  ')
Resultfile.write(accuracy5)
Resultfile.write('\n')

sensitivity1 = TP
sensitivity2 = TP+FN
sensitivity3 = sensitivity1/sensitivity2
print('True positive rate(Sensitivity) =  ',sensitivity3)
sensitivity4 = str(sensitivity3)
Resultfile.write('True positive rate(Sensitivity) =  ')
Resultfile.write(sensitivity4)
Resultfile.write('\n')
#Resultfile.close()

true_mcc = TP*TN
false_mcc = FP*FN
mcc1 = TP+FN
mcc2 = TP+FP
mcc3 = TN+FP
mcc4 = TN+FN
mcc5 = mcc1*mcc2*mcc3*mcc4
mcc6 = math.sqrt(mcc5)
mcc7 = true_mcc - false_mcc
MCC = mcc7/mcc6
MCC1 = str(MCC)
print("MCC = ",MCC)
Resultfile.write('MCC =  ')
Resultfile.write(MCC1)
Resultfile.write('\n')



# creat graph between score and fdr of all cluster of pass and fail PSM
cluster0_match_files_with_all_parameter = 'cluster0_match_with_fdr_result_all_paremeter.csv'
cluster0_notmatch_files_with_all_parameter = 'cluster0_not_match_with_fdr_result_all_paremeter.csv'
cluster1_match_files_with_all_parameter = 'cluster1_match_with_fdr_result_all_paremeter.csv'
cluster1_notmatch_files_with_all_parameter = 'cluster1_not_match_with_fdr_result_all_paremeter.csv'


cluster0_match_files_with_all_parameter = pd.read_csv(cluster0_match_files_with_all_parameter)
cluster0_match_files_with_all_parameter1= cluster0_match_files_with_all_parameter['MSGF_RawScore']
cluster0_match_files_with_all_parameter2= cluster0_match_files_with_all_parameter['q-value']

cluster0_notmatch_files_with_all_parameter = pd.read_csv(cluster0_notmatch_files_with_all_parameter)
cluster0_notmatch_files_with_all_parameter1= cluster0_notmatch_files_with_all_parameter['MSGF_RawScore']
cluster0_notmatch_files_with_all_parameter2= cluster0_notmatch_files_with_all_parameter['q-value']

cluster1_match_files_with_all_parameter = pd.read_csv(cluster1_match_files_with_all_parameter)
cluster1_match_files_with_all_parameter1= cluster1_match_files_with_all_parameter['MSGF_RawScore']
cluster1_match_files_with_all_parameter2= cluster1_match_files_with_all_parameter['q-value']


cluster1_notmatch_files_with_all_parameter = pd.read_csv(cluster1_notmatch_files_with_all_parameter)
cluster1_notmatch_files_with_all_parameter1= cluster1_notmatch_files_with_all_parameter['MSGF_RawScore']
cluster1_notmatch_files_with_all_parameter2= cluster1_notmatch_files_with_all_parameter['q-value']


concanatedfile = pd.concat([cluster0_match_files_with_all_parameter1, cluster0_match_files_with_all_parameter2, cluster0_notmatch_files_with_all_parameter1, cluster0_notmatch_files_with_all_parameter2, cluster1_match_files_with_all_parameter1, cluster1_match_files_with_all_parameter2, cluster1_notmatch_files_with_all_parameter1, cluster1_notmatch_files_with_all_parameter2], axis=1)
concanatedfile.columns =['cluster0_match_MSGF_RawScore', 'cluster0_match_q-value', 'cluster0_notmatch_MSGF_RawScore', 'cluster0_notmatch_q-value', 'cluster1_match_MSGF_RawScore', 'cluster1_match_q-value', 'cluster1_notmatch_MSGF_RawScore', 'cluster1_notmatch_q-value']
concanatedfile.to_csv('dencity_cluster_with_meanerrorall_file_for_graph.csv', index=False)


d = pd.read_csv('dencity_cluster_with_meanerrorall_file_for_graph.csv')
cluster0_match_MSGF_RawScore = d['cluster0_match_MSGF_RawScore']
cluster0_match_qvalue = d['cluster0_match_q-value']
cluster0_notmatch_MSGF_RawScore = d['cluster0_notmatch_MSGF_RawScore']
cluster0_notmatch_qvalue = d['cluster0_notmatch_q-value']
cluster1_match_MSGF_RawScore = d['cluster1_match_MSGF_RawScore']
cluster1_match_qvalue = d['cluster1_match_q-value']
cluster1_notmatch_MSGF_RawScore = d['cluster1_notmatch_MSGF_RawScore']
cluster1_notmatch_qvalue = d['cluster1_notmatch_q-value']

legend = ['cluster0_match', 'cluster0_notmatch','cluster1_match','cluster1_notmatch']


plt.scatter(cluster0_match_qvalue, cluster0_match_MSGF_RawScore, c='Black', alpha=0.5)
plt.scatter(cluster0_notmatch_qvalue, cluster0_notmatch_MSGF_RawScore, c='red', alpha=0.5)
plt.scatter(cluster1_match_qvalue, cluster1_match_MSGF_RawScore, c='green', alpha=0.5)
plt.scatter(cluster1_notmatch_qvalue, cluster1_notmatch_MSGF_RawScore, c='orange', alpha=0.5)
plt.xlabel("qvalue")
plt.ylabel("score")
plt.legend(legend)
#plt.xticks(range(-90, 250))
#plt.yticks(range(1, 20000))
plt.title('score_frequency')
plt.show()




#combine cluster 0 file and check PSM and peptide and fdr
cluster0_match = 'cluster0_match_with_fdr_result_all_paremeter.csv'
cluster0_notmatch= 'cluster0_not_match_with_fdr_result_all_paremeter.csv'
df1 = pd.read_csv(cluster0_match)
df2 = pd.read_csv(cluster0_notmatch)
frames = pd.concat([df1,df2], axis=0)
frames.to_csv('dencity_cluster0_result_match_and_not_match.csv', index=False)
dencity_cluster0_result_match_and_not_match= pd.read_csv('dencity_cluster0_result_match_and_not_match.csv')
dencity_cluster0_result_match_and_not_match.sort_values(["Protein"], axis=0, ascending=True, inplace=True) 
#dencity_cluster0_result_match_and_not_match.to_csv('dencity_cluster0_result_match_and_not_match_short.csv', index=False)
protein_colunm = dencity_cluster0_result_match_and_not_match['Protein']
Contaminant = 0
for i in protein_colunm:
	protein_name = i.split(']')
	if '[Contaminant' in protein_name:
		#print(i)
		Contaminant+=1
	else:
		pass
print('Contaminant =',Contaminant)

GI = 0
for i in protein_colunm:
	protein_name = i.split('|')
	if 'gi' in protein_name:
		#print(i)
		GI+=1
	else:
		pass
print('GI=',GI)

SP = 0
for i in protein_colunm:
	protein_name = i.split('|')
	if 'sp' in protein_name:
		#print(i)
		SP+=1
	else:
		pass
print('SP',SP)

Total_PSM = Contaminant+GI+SP
print('Total_PSM',Total_PSM)
actual_FDR = (2*GI*100)/Total_PSM
Resultfile.write('---------------------------------------------------------------------------------------------')
print('actual_FDR',actual_FDR)
Total_PSM = str(Total_PSM)
Resultfile.write('Total_PSM=')
Resultfile.write(Total_PSM)
Resultfile.write('\n')
Contaminant = str(Contaminant)
Resultfile.write('Contaminant =')
Resultfile.write(Contaminant)
Resultfile.write('\n')
GI = str(GI)
Resultfile.write('GI=')
Resultfile.write(GI)
Resultfile.write('\n')
SP = str(SP)
Resultfile.write('SP = ')
Resultfile.write(SP)
Resultfile.write('\n')
actual_FDR = str(actual_FDR)
Resultfile.write('actual_FDR = ')
Resultfile.write(actual_FDR)
Resultfile.write('\n')


#Compare pass PSM with cluster PSM
print('\n\nMatch actual PSM file with cluster file at PSM Level')
Pass_PSM_file = pd.read_csv(spectrafidfile)
cluster_file = pd.read_csv('dencity_cluster0_result_match_and_not_match.csv')
Pass_PSM_file_sp = Pass_PSM_file[Pass_PSM_file['protein_type'] == 'sp']
#print(Pass_PSM_file_sp)
Pass_PSM_file_gi = Pass_PSM_file[Pass_PSM_file['protein_type'] == 'gi']
#print(Pass_PSM_file_gi)
Pass_PSM_file_Contaminant = Pass_PSM_file[Pass_PSM_file['protein_type'] == 'Contaminant']
#print(Pass_PSM_file_Contaminant)

cluster_file_sp = cluster_file[cluster_file['protein_type'] == 'sp']
#print(cluster_file_sp)
cluster_file_gi = cluster_file[cluster_file['protein_type'] == 'gi']
#print(cluster_file_gi)
cluster_file_Contaminant = cluster_file[cluster_file['protein_type'] == 'Contaminant']
#print(cluster_file_Contaminant)

number_of_sp = pd.merge(Pass_PSM_file_sp, cluster_file_sp, how='inner', on='spectrumID')
number_of_sp_len = len(number_of_sp)
print('number_of_common_sp = ',number_of_sp_len)

number_of_sp_in_pass_PSM = pd.merge(Pass_PSM_file_sp, cluster_file_sp, how='left', on='spectrumID')
number_of_sp_in_pass_PSM_1 = abs(number_of_sp_len - len(number_of_sp_in_pass_PSM))
print('number_of_sp_in_pass_PSM_1 = ',number_of_sp_in_pass_PSM_1)

number_of_sp_in_cluster_PSM = pd.merge(Pass_PSM_file_sp, cluster_file_sp, how='right', on='spectrumID')
number_of_sp_in_cluster_PSM_1 = abs(number_of_sp_len - len(number_of_sp_in_cluster_PSM))
print('number_of_sp_in_cluster_PSM_1 = ',number_of_sp_in_cluster_PSM_1)

print('sp = cluster result(',number_of_sp_in_cluster_PSM_1,number_of_sp_len,number_of_sp_in_pass_PSM_1,')Pass PSM')

number_of_gi = pd.merge(Pass_PSM_file_gi, cluster_file_gi, how='inner', on='spectrumID')
number_of_gi_len = len(number_of_gi)
print('\nnumber_of_common_gi = ',number_of_gi_len)

number_of_gi_in_pass_PSM = pd.merge(Pass_PSM_file_gi, cluster_file_gi, how='left', on='spectrumID')
number_of_gi_in_pass_PSM_1 = abs(number_of_gi_len - len(number_of_gi_in_pass_PSM))
print('number_of_gi_in_pass_PSM_1 = ',number_of_gi_in_pass_PSM_1)

number_of_gi_in_cluster_PSM = pd.merge(Pass_PSM_file_gi, cluster_file_gi, how='right', on='spectrumID')
number_of_gi_in_cluster_PSM_1 = abs(number_of_gi_len - len(number_of_gi_in_cluster_PSM))
print('number_of_gi_in_cluster_PSM_1 = ',number_of_gi_in_cluster_PSM_1)

print('gi = cluster result(',number_of_gi_in_cluster_PSM_1,number_of_gi_len,number_of_gi_in_pass_PSM_1,')Pass PSM')

number_of_Contaminant = pd.merge(Pass_PSM_file_Contaminant, cluster_file_Contaminant, how='inner', on='spectrumID')
number_of_Contaminant_len = len(number_of_Contaminant)
print('\nnumber_of_common_Contaminant = ',number_of_Contaminant_len)

number_of_Contaminant_in_pass_PSM = pd.merge(Pass_PSM_file_Contaminant, cluster_file_Contaminant, how='left', on='spectrumID')
number_of_Contaminant_in_pass_PSM_1 = abs(number_of_Contaminant_len - len(number_of_Contaminant_in_pass_PSM))
print('number_of_Contaminant_in_pass_PSM_1 = ',number_of_Contaminant_in_pass_PSM_1)

number_of_Contaminant_in_cluster_PSM = pd.merge(Pass_PSM_file_Contaminant, cluster_file_Contaminant, how='right', on='spectrumID')
number_of_Contaminant_in_cluster_PSM_1 = abs(number_of_Contaminant_len - len(number_of_Contaminant_in_cluster_PSM))
print('number_of_Contaminant_in_cluster_PSM_1 = ',number_of_Contaminant_in_cluster_PSM_1)

print('Contaminant = cluster result(',number_of_Contaminant_in_cluster_PSM_1,number_of_Contaminant_len,number_of_Contaminant_in_pass_PSM_1,')Pass PSM')


#at peptide level
print('\n\nMatch actual PSM file with cluster file at peptide Level')
Pass_PSM_file_remove_duplicate = pd.read_csv(spectrafidfile)
#print('1',Pass_PSM_file_remove_duplicate)
Pass_PSM_file_remove_duplicate1 = Pass_PSM_file_remove_duplicate.drop_duplicates(subset='Modifiedpeptide')
#print('2',Pass_PSM_file_remove_duplicate1)
cluster_file_remove_duplicate = pd.read_csv('dencity_cluster0_result_match_and_not_match.csv')
cluster_file_remove_duplicate = cluster_file_remove_duplicate.drop_duplicates(subset='Modifiedpeptide')
#print(cluster_file_remove_duplicate)

Pass_PSM_file_remove_duplicate1_sp = Pass_PSM_file_remove_duplicate1[Pass_PSM_file_remove_duplicate1['protein_type'] == 'sp']
#print(Pass_PSM_file_remove_duplicate1_sp)

Pass_PSM_file_remove_duplicate1_gi = Pass_PSM_file_remove_duplicate1[Pass_PSM_file_remove_duplicate1['protein_type'] == 'gi']
#print(Pass_PSM_file_gi)
Pass_PSM_file_remove_duplicate1_Contaminant = Pass_PSM_file_remove_duplicate1[Pass_PSM_file_remove_duplicate1['protein_type'] == 'Contaminant']
#print(Pass_PSM_file_Contaminant)

cluster_file_remove_duplicate_sp = cluster_file_remove_duplicate[cluster_file_remove_duplicate['protein_type'] == 'sp']
#print(cluster_file_sp)
cluster_file_remove_duplicate_gi = cluster_file_remove_duplicate[cluster_file_remove_duplicate['protein_type'] == 'gi']
#print(cluster_file_gi)
cluster_file_remove_duplicate_Contaminant = cluster_file_remove_duplicate[cluster_file_remove_duplicate['protein_type'] == 'Contaminant']
#print(cluster_file_Contaminant)


after_remove_duplicate_number_of_sp = pd.merge(Pass_PSM_file_remove_duplicate1_sp, cluster_file_remove_duplicate_sp, how='inner', on='spectrumID')
after_remove_duplicate_number_of_sp_len = len(after_remove_duplicate_number_of_sp)
print('after_remove_duplicate_number_of_sp = ',after_remove_duplicate_number_of_sp_len)

after_remove_duplicate_number_of_sp_in_pass_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_sp, cluster_file_remove_duplicate_sp, how='left', on='spectrumID')
after_remove_duplicate_number_of_sp_in_pass_PSM_1 = abs(after_remove_duplicate_number_of_sp_len - len(after_remove_duplicate_number_of_sp_in_pass_PSM))
print('after_remove_duplicate_number_of_sp_in_pass_PSM = ',after_remove_duplicate_number_of_sp_in_pass_PSM_1)

after_remove_duplicate_number_of_sp_in_cluster_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_sp, cluster_file_remove_duplicate_sp, how='right', on='spectrumID')
after_remove_duplicate_number_of_sp_in_cluster_PSM_1 = abs(after_remove_duplicate_number_of_sp_len - len(after_remove_duplicate_number_of_sp_in_cluster_PSM))
print('number_of_sp_in_cluster_PSM = ',after_remove_duplicate_number_of_sp_in_cluster_PSM_1)

print('sp = cluster result(',after_remove_duplicate_number_of_sp_in_cluster_PSM_1,after_remove_duplicate_number_of_sp_len,after_remove_duplicate_number_of_sp_in_pass_PSM_1,')Pass PSM')

after_remove_duplicate_number_of_gi = pd.merge(Pass_PSM_file_remove_duplicate1_gi, cluster_file_remove_duplicate_gi, how='inner', on='spectrumID')
after_remove_duplicate_number_of_gi_len = len(after_remove_duplicate_number_of_gi)
print('\nafter_remove_duplicate_number_of_common_gi = ',after_remove_duplicate_number_of_gi_len)

after_remove_duplicate_number_of_gi_in_pass_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_gi, cluster_file_remove_duplicate_gi, how='left', on='spectrumID')
after_remove_duplicate_number_of_gi_in_pass_PSM_1 = abs(after_remove_duplicate_number_of_gi_len - len(after_remove_duplicate_number_of_gi_in_pass_PSM))
print('after_remove_duplicate_number_of_gi_in_pass_PSM_1 = ',after_remove_duplicate_number_of_gi_in_pass_PSM_1)

after_remove_duplicate_number_of_gi_in_cluster_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_gi, cluster_file_remove_duplicate_gi, how='right', on='spectrumID')
after_remove_duplicate_number_of_gi_in_cluster_PSM_1 = abs(after_remove_duplicate_number_of_gi_len - len(after_remove_duplicate_number_of_gi_in_cluster_PSM))
print('after_remove_duplicate_number_of_gi_in_cluster_PSM_1 = ',after_remove_duplicate_number_of_gi_in_cluster_PSM_1)

print('gi = cluster result(',after_remove_duplicate_number_of_gi_in_cluster_PSM_1,after_remove_duplicate_number_of_gi_len,after_remove_duplicate_number_of_gi_in_pass_PSM_1,')Pass PSM')

after_remove_duplicate_number_of_Contaminant = pd.merge(Pass_PSM_file_remove_duplicate1_Contaminant, cluster_file_remove_duplicate_Contaminant, how='inner', on='spectrumID')
after_remove_duplicate_number_of_Contaminant_len = len(after_remove_duplicate_number_of_Contaminant)
print('\nafter_remove_duplicate_number_of_common_Contaminant = ',after_remove_duplicate_number_of_Contaminant_len)

after_remove_duplicate_number_of_Contaminant_in_pass_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_Contaminant, cluster_file_remove_duplicate_Contaminant, how='left', on='spectrumID')
after_remove_duplicate_number_of_Contaminant_in_pass_PSM_1 = abs(after_remove_duplicate_number_of_Contaminant_len - len(after_remove_duplicate_number_of_Contaminant_in_pass_PSM))
print('after_remove_duplicate_number_of_Contaminant_in_pass_PSM_1 = ',after_remove_duplicate_number_of_Contaminant_in_pass_PSM_1)

after_remove_duplicate_number_of_Contaminant_in_cluster_PSM = pd.merge(Pass_PSM_file_remove_duplicate1_Contaminant, cluster_file_remove_duplicate_Contaminant, how='right', on='spectrumID')
after_remove_duplicate_number_of_Contaminant_in_cluster_PSM_1 = abs(after_remove_duplicate_number_of_Contaminant_len - len(after_remove_duplicate_number_of_Contaminant_in_cluster_PSM))
print('after_remove_duplicate_number_of_Contaminant_in_cluster_PSM_1 = ',after_remove_duplicate_number_of_Contaminant_in_cluster_PSM_1)

print('Contaminant = cluster result(',after_remove_duplicate_number_of_Contaminant_in_cluster_PSM_1,after_remove_duplicate_number_of_Contaminant_len,after_remove_duplicate_number_of_Contaminant_in_pass_PSM_1,')Pass PSM')

os.remove('result.csv')
os.remove(files+'_ result_cluster0.csv')
os.remove(files+'_result_cluster1.csv')
os.remove(clusterfile0+"_onlyspectra_result.csv")
os.remove(clusterfile1+"_onlyspectra_result.csv")
os.remove(clusterfile0+'_cluster0_match_with_fdr_result.csv')
os.remove(clusterfile0+'_not_match_with_fdr_result_result.csv')
os.remove(clusterfile1+'_cluster1_match_with_fdr_result.csv')
os.remove(clusterfile1+'_not_match_with_fdr_result_result.csv')
os.remove('cluster0_match_with_fdr_result_all_paremeter.csv')
os.remove('cluster0_not_match_with_fdr_result_all_paremeter.csv')
os.remove('cluster1_match_with_fdr_result_all_paremeter.csv')
os.remove('cluster1_not_match_with_fdr_result_all_paremeter.csv')
os.remove('dencity_cluster_with_meanerrorall_file_for_graph.csv')
os.remove("spectrumid"+fdrspectrafile)
os.remove('dencity_cluster0_result_match_and_not_match.csv')
os.remove(spectrafidfile)

