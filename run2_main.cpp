

//#include<cstdio>
#include<opencv2\opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vl/generic.h>
#include <vl/kmeans.h>
#include <stdint.h>


using namespace std;
using namespace cv;
using namespace flann;

// location of the traning images folder:
#define TRAINING_PATH "../../Coursework_3/training/"
// patch window size & offset:
#define PATCH_SIZE 8
#define PATCH_OFFSET (PATCH_SIZE/2)
#define DICTIONARY_SIZE  500
#define NR_OF_TR_FILES 100     // number of files to use for training the linear classifiers
#define NR_OF_PATCH_TR_FILES 5 // number of files for the vocabulary
#define MAX_TRAINING_ITER 500  // maximum training iterations for the linear classifiers

#define DIMENSIONS (PATCH_SIZE*PATCH_SIZE) // Nr. of dimensions of each patch descriptor. DO NOT CHANGE!

/* set to 0 if vocabulary doesnt exists or if it's not available yet*/
#define VOCABULARY_EXISTS 1
/* set to 0 if BoVW features are not yet extracted*/
#define FEATURES_EXTRACTED 1
/* set to 0 if classifiers are not trained */
#define TRAINED 0
/* set to 1 to classify */
#define CLASSIFY 0

const string folder[]={
			"bedroom","Coast","Forest","Highway",
			"industrial","Insidecity","kitchen","livingroom",
			"Mountain","Office","OpenCountry","store",
			"Street","Suburb","TallBuilding"};

/* Debugging reminder: LoC with possible faults are followed by the comment: [xF],
  where 'x' is the fault index.
  Till now:
  [1F]- a little bit 'obscure' function. Is scaling necessary, 
		considering normalisation?!
  [2F]- a possible normalising error (terminology conflict!).
  */
/* Mean-center, normalise and concatenate patch into a vector. */
int NormalisePatch(Mat& patch)
{

					//cout<<"8x8:\n"<<patch<<endl;
		// 1. convert it to 1-by-(PATCH_SIZE^2) array
			if(patch.isContinuous()) 				
				patch=patch.reshape(0,1); 
			else 
				{ cout<<"Error: patch is not continuous!\n"; return -2; }
					//cout<<"1x64:\n"<<patch<<endl;
		// 2. convert to float:
			patch.convertTo(patch, CV_32F); // !!! possible fault !!! [1F]
					//cout<<"float:\n"<<patch<<endl;
		// 3. mean centre:
			patch=patch-mean(patch);
					//cout<<"meanC:\n"<<patch<<endl;
		// 4. normalise:   !!! possible fault !!! [2F]
					//cout<<"norm:\n"<<norm(patch,NORM_L2)<<endl;
			patch=patch/norm(patch,NORM_L2);
			//!!! normalize(patch, patch, 0, 1, NORM_MINMAX, CV_32F);
					//cout<<"normal:\n"<<patch<<endl;
					//cout<<"norm:\n"<<norm(patch,NORM_L2)<<endl;
					//system("pause");
		return 0;
}


/* Generates a bag of Visual words decriptor from a given image*/
int generate_BoW(Mat& img,Mat& Descript,Mat& vocabulary)
{
    
	// create nearest neighbor search index and its parameters:
	  // random Kdtree search:
	  //KDTreeIndexParams iParams;
	  // brute force linear search: 
	  KDTreeIndexParams iParams; 
	Index KDTSearch(vocabulary,iParams);

	Mat patch,indices,dist;
	Descript=Mat::zeros(1, DICTIONARY_SIZE, CV_32S);
	float radius=50000.0;
	int retErr;
	//int index;
	for(int i=PATCH_OFFSET; i<=img.rows-PATCH_OFFSET; i+=PATCH_OFFSET)
		for(int j=PATCH_OFFSET; j<=img.cols-PATCH_OFFSET; j+=PATCH_OFFSET)
		{
		// extract patch:
			img(Rect(j-PATCH_OFFSET,i-PATCH_OFFSET,PATCH_SIZE,PATCH_SIZE)).copyTo(patch); // !!!! put i,j !
					//cout<<"8x8:\n"<<patch<<endl;
		    
			// mean centre, normalise and concatenate rows: 
			retErr=NormalisePatch(patch);

			if(retErr!=0) return retErr;
					//cout<<"1x64:\n"<<patch<<endl;
			// find nearest neighbour:
			//LinSearch.radiusSearch(patch,indices,dist,radius,1,SearchParams(500,0.00001,false));
			KDTSearch.knnSearch(patch,indices,dist,1,SearchParams(32,0,false));
					//cout<<"indices:"<<indices.at<int>(0)<<endl;
					//cout<<"dist= "<<dist<<endl;
			
		    Descript.at<int>(indices.at<int>(0))++;
					//system("pause");

		}
		return 0;
}	

/**/


int main()
{
	int count=0;
	string filepath;
	double percentDone=0;
	int retErr;
	Mat src,dictionary;
#if VOCABULARY_EXISTS == 0	
	Mat patch,UnclusteredData;
	/* peform necessary steps to build vocabulary */
// 1.---------- Gather patches: ------------------------------	
	
	
	cout.precision(3);
	
	cout<<"Step 1: Getting patches from training data:\n";

	for(int scene=0;scene<15;scene++)
	for(int fileNr=0; fileNr<NR_OF_PATCH_TR_FILES;fileNr++)
	{
		count++;
		filepath=TRAINING_PATH + folder[scene]+"/"+to_string(fileNr)+".jpg";
	
		src=imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
			if(! src.data ) // Check for invalid image
			{
				cout << "Error: Could not open or find the image!\n";
				return -1;
			}
			//imshow("image",src);
			
		/* !!! NOTE: we are intentionally leaving small sized patches in the 
		right and bottom borders uncovered, because their size would be less 
		than half of normal patches! */
		for(int i=PATCH_OFFSET; i<=src.rows-PATCH_OFFSET; i+=PATCH_OFFSET)
		for(int j=PATCH_OFFSET; j<=src.cols-PATCH_OFFSET; j+=PATCH_OFFSET)
		{
		// extract patch:
			src(Rect(j-PATCH_OFFSET,i-PATCH_OFFSET,PATCH_SIZE,PATCH_SIZE)).copyTo(patch); // !!!! put i,j !
					//cout<<"8x8:\n"<<patch<<endl;
		    
			// mean centre, normalise and concatenate rows: 
			retErr=NormalisePatch(patch);
			if(retErr!=0) return retErr;
			// insert to unclustered data matrix:
			UnclusteredData.push_back(patch); 
			//count++;
		}
		percentDone=(double)count/(NR_OF_PATCH_TR_FILES*0.15);
		cout<<"\r"<<percentDone<<fixed<<"%   ";
	}
	    
		cout<<"\nDone!\nUnclustered data size: "<<UnclusteredData.size()<<endl;
// ------------- End of GATHER PATCHES. ----------------------------------------


// 2.--------------- Cluster data: ------------------------------------------------
	cout<<"Step 2: Clustering with "<<DICTIONARY_SIZE<<" centers...\n";
	
	//Create a Bag of word (BoW or BoF) trainer:
	double err;	
	
	Mat labels;
	int attempts=3;
	
	//int clustercount=DICTIONARY_SIZE;
   /* NOTHING WORKS HERE, OPENCV IS MISERABLE!!!
     Does not work for huge dimensional data
    
		// terminate after 10000 iterations or 0.00001 accuracy:	
        TermCriteria crit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.00001);
    
  
	//BOWKMeansTrainer BOW(DICTIONARY_SIZE,crit,3,KMEANS_PP_CENTERS);
	//dictionary=BOW.cluster(UnclusteredData);
	//cout<<"unclust:\n"<<UnclusteredData<<endl;
	err=kmeans(UnclusteredData, clustercount,labels,crit,attempts, KMEANS_RANDOM_CENTERS, dictionary);
	
	/*End of "NOTHING WORKS HERE, OPENCV IS MISERABLE!!!"*/
	
    /*Second method clustering using VLFeat: */

	// 1.set float data and euclidean distance:
    VlKMeans * kmeans1= vl_kmeans_new (VL_TYPE_FLOAT,VlDistanceL2);
	// choose initialisation algorithm
	vl_kmeans_set_initialization(kmeans1,VlKMeansPlusPlus);

	// 2.choose algorithm type= Lloyd :
	vl_kmeans_set_algorithm( kmeans1,VlKMeansLloyd);

	// 3.convert data to float
	float* Unclust= (float*)UnclusteredData.data, *Dict;

	// init centers with kmean++:
	//vl_kmeans_init_centers_plus_plus(kmeans1,Unclust,DIMENSIONS,UnclusteredData.rows,DICTIONARY_SIZE);

	// 4.set stop criteria=100 iterations:
	 vl_kmeans_set_num_repetitions(kmeans1,2);
	// 5.Cluster or refine centers:
		//vl_kmeans_refine_centers(kmeans1,Unclust,UnclusteredData.rows);
	 vl_kmeans_cluster(kmeans1,Unclust,DIMENSIONS,UnclusteredData.rows,DICTIONARY_SIZE);
	// 6.obtain energy (error measurement):
	err = vl_kmeans_get_energy(kmeans1);

	// get centers:
	Dict = (float*)vl_kmeans_get_centers(kmeans1);
	// copy to Matrix:
	    /* 1st method:*/
		memcpy(dictionary.data,Dict,DICTIONARY_SIZE*DIMENSIONS*sizeof(float));
		/*---END M1---*/

		/* 2nd method:*
		for(int i=0;i<DIMENSIONS;i++)
		for(int j=0;j<DICTIONARY_SIZE;j++)
			dictionary.at<float>(j,i)=Dict[(j*DIMENSIONS)+i];
		/*---END M2---*/

	/*End of Second method clustering using VLFeat: */
	cout<<*Dict<<endl;

	// write result to files:
	FileStorage f1("dictionary.yml", FileStorage::WRITE);	
	f1 << "dictionary" << dictionary;
	f1.release();

	//FileStorage f2("UnclusteredData.yml", FileStorage::WRITE);
	//f2<< "Unclustered patches" << UnclusteredData;
	//f2.release();

	cout<<"dictionary size:\n"<<dictionary.size()<<endl;
	cout.precision(10);
	cout<<"energy: "<<err<<endl;
	//cout<<"dictionary:\n"<<dictionary<<endl;
    cout<<"Done!\n";
	cout<<"Continuity:\n"<<((dictionary.isContinuous())?"yes":"no")<<endl;
// --------------- END of Cluster data: ------------------------------------------------
#endif 

    Mat descriptor,Descriptors;
	//cout<<"zero decriptor"<<descriptor<<endl;
				//cout<<"descriptor size: "<<  descriptor.size()<<endl;
				//descriptor.at<uint16_t>(7)++;
#if (FEATURES_EXTRACTED == 0)
	/* extract BoVW fetures from training images */
	cout<<"Step 3: Extracting BoVW features from training images:\n";			
	// read dictionary:
	FileStorage fd("dictionary.yml", FileStorage::READ);
    fd["dictionary"] >> dictionary;
    fd.release();    
	cout.precision(3);
		cout<<"dictionary size: "<< dictionary.size()<<endl;
				//cout<<"dictionary: "<< dictionary.row(499)<<endl;
    // step
	count=0;
	for(int scene=0;scene<15;scene++)
	for(int fileNr=0; fileNr<NR_OF_TR_FILES;fileNr++)
	{
		count++;
		filepath=TRAINING_PATH + folder[scene]+"/"+to_string(fileNr)+".jpg";
		src= imread(filepath,CV_LOAD_IMAGE_GRAYSCALE);
		retErr=generate_BoW(src,descriptor,dictionary);
		if(retErr!=0) return retErr;
		// add label:
		hconcat(descriptor,Mat(1,1,CV_32S,scene),descriptor);
				//if(!(scene || fileNr)) cout<<"decriptor + label size= "<<descriptor.size()<<endl;
		Descriptors.push_back(descriptor);		
				//if(fileNr==0){cout<<"BoVW decriptor"<<descriptor<<endl; system("pause");}
		percentDone=(double)count/(NR_OF_TR_FILES*0.15);
		cout<<"\r"<<percentDone<<fixed<<"%   ";
	}
	cout<<"Descriptors size:"<<Descriptors.size()<<endl;
	FileStorage ff("BoVWfeatures.yml", FileStorage::WRITE);
    ff<<"Descriptors"<< Descriptors;
    ff.release();  
	cout<<"Done!\n";
#endif

#if (TRAINED == 0) 
	/* create and train the classifiers */
	FileStorage ffd("BoVWfeatures.yml", FileStorage::READ);
    ffd["Descriptors"] >> Descriptors;
    ffd.release();
	cout<<"Step 4: Training the classifiers:\n";	
	cout<<"Loaded 1st row descriptors size:"<<Descriptors.row(1).size()<<endl;
	// 1. copy labels and add to descriptor matrix by a column of ones:
	Mat labels;
	Descriptors.col(DICTIONARY_SIZE).copyTo(labels);
				//cout<<"labels: "<<labels<<endl;
				//system("pause");
	Descriptors.col(DICTIONARY_SIZE).setTo(1.0);
		Descriptors.convertTo(Descriptors,CV_32F);
				//cout<<"D.ones: "<<Descriptors.col(DICTIONARY_SIZE)<<endl;
				//system("pause");
	// 2. shuffle indices:
    vector<int> indx;
		for(int i=0;i<Descriptors.rows;i++) indx.push_back(i);
	randShuffle(indx);
				//FileStorage test("indx.yml", FileStorage::WRITE);
				//test<<"indx"<< indx;
				//test.release();  
				//system("pause");
	// 3. Create matrix with hyperplane coefficients:
	Mat A(15,DICTIONARY_SIZE+1,CV_32F,0.0);
				//cout<<"zA= "<<A<<endl;
	  	// initialize A with random uniform numbers:
		randu(A,-1.0,1.0);
				//cout<<"A= "<<A<<endl;
				cout<<"'Asize= "<<A.t().size()<<endl;
	// 4. start learning:
	bool completed=false; //,Trained[15]={false};
	Mat tClassification;
	float learningRate=0.1;	
	int iter=0;
#define IND indx[i] // shuffled index
	/* class1: */
	// !!!!!!!!!!!!!! TODO complete learning !!!!!!!!!!!!!
	while(!completed && iter<MAX_TRAINING_ITER)
	{
		completed=true;
		iter++;
		count=0;
		cout<<"Iteration\t"<<iter<<":\n";
		// iterate across each image descriptor:
		for(int i=0;i<Descriptors.rows;i++)
		{
			// predict:
			tClassification=A*Descriptors.row(IND).t();
									//cout<<"C= \n"<<tClassification<<endl;
									//cout<<"L= \n"<<labels.at<int>(IND)<<endl;
					
			/* check acurracy of prediction for every category:
			 if prediction is >= 0, then the descriptor belongs to that scene. */
			for(int scene=0;scene<15;scene++)
			{
				if(scene==labels.at<int>(IND) )
				{
					if (tClassification.at<float>(scene)<0 )
				    {
					    
									//cout<<"scene "<<scene<<" is wrong\n"; system("pause");
					    /* correct the respective hyperplane: */
						completed=false;
									//cout<<"A':\n"<<A.row(scene).t()<<endl;
									//system("pause");
						A.row(scene)= A.row(scene) + (learningRate*Descriptors.row(IND));
									//cout<<"A' corrected:\n"<<A.row(scene).t()<<endl;
									//system("pause");
				    }
				}else
				if (tClassification.at<float>(scene)>=0 )
				{
					    
								//cout<<"scene "<<scene<<" is wrong\n"; system("pause");
					/* correct the respective hyperplane: */
					completed=false;
								//cout<<"A':\n"<<A.row(scene).t()<<endl;
								//system("pause");
					A.row(scene)= A.row(scene) - (learningRate*Descriptors.row(IND));
								//cout<<"A' corrected:\n"<<A.row(scene).t()<<endl;
								//system("pause");
				}

			}
			count++;
			cout<<"\r\t"<<(double)100*count/Descriptors.rows<<fixed<<"%   ";
			
		}
		cout<<endl;
	}
	FileStorage ft("Trained_Hyperplanes.yml", FileStorage::WRITE);
    ft<<"A"<< A;
    ft.release();  
	cout<<"Done!\n";

#endif

#if CLASSIFY == 1



#endif
	//namedWindow( "image", WINDOW_NORMAL ); 
	//imshow("image",src);
	//imshow("patch1",src(Rect(src.cols-50,src.rows-50,50,50)));
	//imshow("patch2",src(Rect(0,4,8,8)));
	//cout<<filepath<<endl;
	
	//cout<<"[rows,cols]= "<<src.rows<<","<<src.cols<<endl;
    waitKey(0);
	system("pause"); // windows only!
	return 1;
}