# Face-Recognition-using-Transfer-Learning
Face Recognition using Tensor Flow and FaceNet.
Goal: To generate a model which recognises the faces, with images given as input.


To get face feature embeddings, we used FaceNet model.
FaceNet is a one-shot model, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. We used embeddings from FaceNet to get features which are further used to predict the class representing face of a particular person.

## PREPARING DATA ##
### Training Data ### 
 Images of 6 persons including me (4 images each) were loaded using matplotlib
 Embedder from FaceNet is used to get the final feature vector.
 Labels for the training data were given manually as an array with values 1,2,3,4,5,6 representing 6 different persons.
 
 ### Testing Data ###
 Testing Data consists of 7 images of persons .
 Features were extracted in the same manner from the images as training dataset.
 
 ### Model ###
 On the extracted feature vector, a multiclass logistic regression was applied to learn a classification model.
 
 **cost function** : Cost(hθ(x),y)=−ylog(hθ(x))−(1−y)log(1−hθ(x)), 
 where   y = y_true and
         hθ(x) =y_predicted .
 ### Prediction ###
 In case the prediction probability is lesser than a given threshold, we say the image is of some unknown 'other' person than    those 6 in training data, else a class number(1-6) is returned as output.
 The test data was applied on the model and a score of 1.0 (i.e 100%) was observed on the test data.
### Justification of Accuracy ###
The FaceNet is trained on a huge dataset of face images, with lot of variation. When we applied it on our small dataset(24 images for 6 persons), all the face features were embedded perfectly. When we applied the multiclass classification on those features, it learnt the variation perfectly resulting in 100% accuracy .
