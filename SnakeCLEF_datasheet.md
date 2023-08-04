# SnakeCLEF datasheet

## Motivation

### For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.
The dataset was created as the dataset to be learnt by algorithms participating in the 2023 SnakeCLEF competition. 
### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?
The dataset was created by Lukas Picek, Marek Hruz and Rail Chamidullin of the University of West Bohemia, Andrew Durso of Florida Gulf Coast University, and Isabelle Bolon of the University of Geneva. The competition it was created for is one of the LifeCLEF competitions, which are organised by the CLEF Initiative.
### Any other comments?
None

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Please provide details.
Each instance corresponds to one photograph of a snake.
### How many instances of each type are in total?
182261 total instances, with the number of instances of each species varying from very small numbers to 2079.
### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).
The instances of the dataset are taken from data uploaded to the iNaturalist and HerpMapper websites. The method by which the dataset was sampled from the original data is unknown to the author of the datasheet.
### What does each instance consist of? Raw data? Unprocessed? Text, images?
Each instance consists of the photograph itself along with information including the species of the snake and which country it was photographed in.
### Are there any labels to the data?
See previous question.
### Is there any missing information from individual instances?
The country is missing from some instances, and some instances in the training set have metadata but no associated image.
### Are relationships between individual instances made explicit?
Each instance is associated with an observation number, with photographs from the same observation being given the same number.
### Are there recommended data splits (e.g. train / test)? Provide a description of the splits, and the rationale behind them.
Yes, the dataset is split into a training set, validation set and public test set.
### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? 
The dataset is self-contained.
### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.
Not to the knowledge of the author of the datasheet, but see “If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?” below.
### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
No.
### Does the dataset identify any subpopulations (e.g., by age, gender)?
The dataset consists of images of snakes, which are only identified by species.
### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.
No.
### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description. 
No.
### Any other comments?
None

## Collection process

### How was the data associated with each instance acquired? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.
The creators of the dataset took the data instances from observations of snakes uploaded by the observers to one of the websites iNaturalist and HerpMapper.
### If the data is a sample of a larger subset, what was the sampling strategy? Deterministic, random, etc...?
Unknown to the author of the datasheet.
### Over what time frame was the data collected?
The data was collected between 1990 and 2023 and uploaded to the internet beginning in 2008.
### Were there any ethical review processes conducted (e.g. by an institutional reviewing board?)
Unknown to the author of the datasheet.
### Were the individuals notified of the collection of the data?
The individuals willingly uploaded the data to iNaturalist and HerpMapper. The author of the datasheet does not know whether the creators of the SnakeCLEF dataset notified the individuals that their data was being included in it.
### Did the individuals consent to their data being collected?
The individuals willingly uploaded the data to iNaturalist and HerpMapper. The author of the datasheet does not know whether the creators of the SnakeCLEF dataset asked for the consent of the individuals.
### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? 
The terms of service of both iNaturalist and HerpMapper appear to suggest that data published on their websites become their property. Therefore, although it may be possible to delete data from either site, they still seem to have a right to use it even after it has been deleted. Whether the SnakeCLEF dataset includes any instances that have been deleted from iNaturalist or HerpMapper is unknown to the author of the datasheet, as is whether the creators of the SnakeCLEF dataset provided individuals with a mechanism to revoke consent for inclusion in their sample.
### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? 
Not to the knowledge of the author of the datasheet.
### Any other comments?
None

## Preprocessing/cleaning/labelling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section. 
The larger images were shrunk to a maximum side length of 240 pixels in order to form the “small size” image data. The details of the shrinking process are unknown to the author of the datasheet.
### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? 
Yes, the full-size images are also available.
### Any other comments?
None

## Uses

### What other tasks could the dataset be used for? 
The dataset could be used to train image generator models which would output pictures of snakes given the name of the species.
### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms? 
The dataset may contain a disproportionate number of instances from observations in developed countries, and so may underestimate the prevalence of some species in less developed areas.
### Are there tasks for which the dataset should not be used? If so, please provide a description.
The dataset should not be used to identify species in poorly explored areas where it is likely that unfamiliar species could exist which would not be present in the dataset.
### Any other comments?
None

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.
The dataset is publicly available.
### How will the dataset be distributed?
The dataset is available on Hugging Face.
### When will the dataset be distributed?
The dataset was made available in 2023.
### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions. 
The copyright status of the dataset is unknown to the author of the datasheet.
### Any other comments?
None

## Maintenance

### Who will be maintaining the dataset?
The dataset was created for a specific competition and so is fixed. It therefore does not need to be maintained, but updated versions of the dataset should appear in future SnakeCLEF competitions.
### Any other comments?
None
