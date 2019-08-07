# Deep_learning_for_android_malware_detection
Effectiveness of additional training of an ANN based model in detecting android malware.

![proposed_scheme](https://user-images.githubusercontent.com/36197370/59243894-db837100-8c44-11e9-8fe3-d9c8d1b2a980.PNG)

**The goal of this project is to show the weakness of an ANN based malware detection model in detecting adversarial samples and how to boost its performance.**


*In this project, I build an ANN based android malware detection model. It is trained and tested on data extracted from both benign and malware samples.*

*Using a GAN, we generate adversarial malware samples that we use to attack the model.*
*To boost the model's performance, we retrain it using adversarial samples.*


# ----------Content Description--------------------------

**Feature_Extraction_&_Selection--Directory**

In *Feature_Selection folder*,  we do feature selection with get_important_features.py and generated final dataset with script3_create_final_combined_dataset_important_features.py.

*Bfeatures_jsons.json*--contains data for 3090 benign samples analysed.

*Mfeatures_jsons.json*--contains data for 3090 malware samples analysed.

*M_Bfeatures_jsons.json*---combined raw data for all samples(Benign and Malware).

*script2_create_combined_dataset_all_features.py* --for generating dataset with all features for each sample(observation).

**Generating adversarial examples with GAN---Directory**

*Code contained here is to generate adversarial malware examples that are used
to test SmartAM1, train SmartAM2, test SmartAM2.*

*Step1*

Create input for GAN using create_input_for_gan.py which loads MSGmalware_analysis_dataset_if.csv and returns dataset_if.npz which is the input for GAN(GAN_4_SmartAM.py).

*Step2*

Execute GAN_4_SmartAM.py. At each run, it returns an adversarial batch e.g. adverV1-8.npz(as in advers directory).

*step3*

In advers directory, execute prepare_adver_dataset.py. Its input is adverVX.npz and output is adver_dataset.csv (adversarial dataset like adver_dataset_if_test_1 and  adver_dataset_if_train_1 that are used in the next phase).

**SmartAM_ANN_Model--Directory**

*Contains original samples dataset(MSGmalware_analysis_dataset_if.csv), adversarial examples datasets for training , and testing(
adver_dataset_if_train_1.csv, adver_dataset_if_test_1.csv), SmartAM1_ANN.py(weak model), and SmartAM2_ANN.py(boosted model).*





