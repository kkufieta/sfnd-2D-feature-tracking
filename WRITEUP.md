# Writeup

## Data Buffer Optimization
Task: Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.

Implementation:
```cpp
int dataBufferSize = 2;       // no. of images which are held in memory (ring
                              // buffer) at the same time
vector<DataFrame> dataBuffer; // list of data frames which are held in memory
                              // at the same time

for (...) {
    if (dataBuffer.size() >= dataBufferSize) {
      dataBuffer.erase(dataBuffer.begin());
    }
}
```

Explanation: If the size of the data buffer exceeds the data buffer size (here: 2), the first element of the data buffer is deleted. This makes sure that the data buffer always holds the two most recent images that are being read in.

## Keypoints
### Keypoint Detection
Task: Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.

Implementation:
```cpp
// extract 2D keypoints from current image
vector<cv::KeyPoint>
    keypoints; // create empty feature list for current image
bool bVis = true;

// Detector types:
// -> Gradient Based: HARRIS, SHITOMASI, SIFT
// -> Binary: BRISK, ORB, AKAZE, FAST
string detectorType = "HARRIS";

if (detectorType.compare("SHITOMASI") == 0) {
    detKeypointsShiTomasi(keypoints, imgGray, bVis);
} else if (detectorType.compare("HARRIS") == 0) {
    detKeypointsHarris(keypoints, imgGray, bVis);
} else { // SIFT, BRISK, ORB, AKAZE, FAST
    detKeypointsModern(keypoints, imgGray, detectorType, bVis);
}
```
* The implementation of the functions `detKeyPointsShiTomasi`, `detKeypointsHarris`, `detKeypointsModern` can be found in the file `matching2D.cpp`


### Keypoint Removal
Task: Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

Implementation:
```cpp
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
vector<cv::KeyPoint> focusedKeypoints;
if (bFocusOnVehicle) {
    for (cv::KeyPoint keypoint : keypoints) {
    if (vehicleRect.contains(keypoint.pt)) {
        focusedKeypoints.push_back(keypoint);
    }
    }
}
keypoints = focusedKeypoints;
```
By using `vehicleRect.contains`, we can check which keypoints are in the rectangle and put them into a new vector of keypoints. Creating a new vector and copying it over to the original `keypoints` vector is more time efficient than removing keypoints in random positions from the `keypoints` vector.

## Descriptors

### Keypoint Descriptors
Task: Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

Implementation:
```cpp
/* EXTRACT KEYPOINT DESCRIPTORS */
cv::Mat descriptors;
string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
descKeypoints((dataBuffer.end() - 1)->keypoints,
                (dataBuffer.end() - 1)->cameraImg, descriptors,
                descriptorType);

// push descriptors for current frame to end of data buffer
(dataBuffer.end() - 1)->descriptors = descriptors;
```

`descKeypoints` is implemented in `matching2D.cpp`: 
```cpp
// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0) {

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for
                               // sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorType.compare("SIFT") == 0) {
    extractor = cv::SIFT::create();
  } else if (descriptorType.compare("ORB") == 0) {
    extractor = cv::ORB::create();
  } else if (descriptorType.compare("AKAZE") ==
             0) { // works only with AKAZE keypoints
    extractor = cv::AKAZE::create();
  } else if (descriptorType.compare("BRIEF") == 0) {
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else if (descriptorType.compare("FREAK") == 0) {
    extractor = cv::xfeatures2d::FREAK::create();
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
       << " ms" << endl;
}
```

### Descriptor Matching
Task: Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.

Implementation:
```cpp
vector<cv::DMatch> matches;
string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
string descriptorType_ = descriptorType.compare("SIFT") == 0
                            ? "DES_HOG"
                            : "DES_BINARY"; // DES_BINARY, DES_HOG
string selectorType = "SEL_KNN";             // SEL_NN, SEL_KNN

matchDescriptors((dataBuffer.end() - 2)->keypoints,
                (dataBuffer.end() - 1)->keypoints,
                (dataBuffer.end() - 2)->descriptors,
                (dataBuffer.end() - 1)->descriptors, matches,
                descriptorType_, matcherType, selectorType);

// store matches in current data frame
(dataBuffer.end() - 1)->kptMatches = matches;
```

FLANN matching and k-nearest-neighbor selection with `k=2` are implemented in `matching2D.cpp`.

### Descriptor Distance Ratio
Task: Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

Implementation:
```cpp
// Implement k-nearest-neighbor matching
vector<vector<cv::DMatch>> knnMatches;
int k = 2;
double t = (double)cv::getTickCount();
matcher->knnMatch(descSource, descRef, knnMatches,
                    2); // finds the 2 best matches
t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
cout << " (kNN) with k=2 and n=" << knnMatches.size() << " matches in "
        << 1000 * t / 1.0 << " ms" << endl;

// Filter matches using descriptor distance ratio test
float ratio;
int numTotalMatches = knnMatches.size();
int numDiscardedMatches = 0;
double minDistanceRatio = 0.8;
for (vector<cv::DMatch> kMatch : knnMatches) {
    ratio = kMatch[0].distance / kMatch[1].distance;
    if (ratio <= minDistanceRatio) {
    matches.push_back(kMatch[0]);
    } else {
    numDiscardedMatches += 1;
    }
}
cout << "Number discarded matches: " << numDiscardedMatches << endl;
cout << "Percentage discarded matches: "
        << numDiscardedMatches / (float)numTotalMatches << endl;
```

## Performance
### Count number of keypoints
Task: Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

| Detector | Stats | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Average | 
|:-- |:-- |:-- |:-- |:-- |:-- |:-- |:-- |:-- |:-- |:-- |:-- | :-- | 
| Harris | # keypoints | 115 | 98 | 113 | 121 | 160 | 388 | 85 | 210 | 171 | 281 | 170.42 |
| | Time [ms] | 12.4 | 13.4 | 7.7 | 6.6 | 7.5 | 7.3 | 7.5 | 7.1 | 7.1 | 7.6 | 8.42 |
| | # selected keypoints | 17 | 14 | 18 | 21 | 26 | 43 | 18 | 31 | 26 | 34 | 24.8 |
| | mean | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 
| | std dev | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 


### Count number of matched keypoints
Task: Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

### Log runtime of algorithms
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.