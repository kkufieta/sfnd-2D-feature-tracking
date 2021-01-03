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
| | avg. keypoint size | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 
| | keypoint size std dev | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| SHITOMASI | # keypoints | 1370 | 1301 | 1361 | 1358 | 1333 | 1284 | 1322 | 1366 | 1389 | 1339 | 1342.3 |
| | Time [ms] | 15.5 | 12.7 | 10.7 | 10.8 | 10.9 | 10.9 | 11.1 | 10.4 | 10.4 | 10 | 11.34 |
| | # selected keypoints | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 | 117.9 |
| | avg. keypoint size | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 
| | keypoint size std dev | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| SIFT | # keypoints | 1438 | 1371 | 1380 | 1335 | 1304 | 1369 | 1369 | 1382 | 1462 | 1422 | 1383.2 |
| | Time [ms] | 190.8 | 169.7 | 161.6 | 141.5 | 140.4 | 136.8 | 137.3 | 140.2 | 135.5 | 138.6 | 149.2 |
| | # selected keypoints | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 | 138 |
| | avg. keypoint size | 4.98 | 5.1 | 4.9 | 4.7 | 4.7 | 4.7 | 5.4 | 4.6 | 5.5 | 5.6 | 5.0 | 
| | keypoint size std dev | 35.2 | 38.1 | 36.2 | 27.5 | 30.4 | 31.1 | 42.4 | 26.5 | 44.5 | 44.6 | 35.7 | 
| BRISK | # keypoints | 2757 | 2777 | 2741 | 2735 | 2757 | 2695 | 2715 | 2628 | 2639 | 2672 | 2711.6 | 
| | Time [ms] | 78.8842 | 64.1708 | 61.6984 | 62.0059 | 61.3036 | 56.1614 | 59.3102 | 61.2397 | 62.2673 | 62.3111 | 62.9353 | 
| | # selected keypoints | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 266 | 254 | 276.2 | 
| | avg. keypoint size | 21.5492 | 21.7853 | 21.6509 | 20.3583 | 22.5911 | 22.9442 | 21.8014 | 22.1472 | 22.5558 | 22.0389 | 21.9422 | 
| | keypoint size std dev | 212.496 | 212.129 | 191.029 | 159.23 | 220.702 | 249.964 | 215.379 | 226.018 | 230.701 | 215.124 | 213.277 | 
| ORB | # keypoints | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 
| | Time [ms] | 118.945 | 7.28081 | 7.31561 | 7.12867 | 7.02634 | 7.22899 | 7.59878 | 7.1308 | 7.30686 | 8.04887 | 18.5011 | 
| | # selected keypoints | 92 | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 | 116.1 | 
| | avg. keypoint size | 57.0723 | 57.2273 | 56.4948 | 55.1436 | 56.7442 | 56.6367 | 56.7683 | 55.4296 | 54.6723 | 54.3885 | 56.0578 | 
| | keypoint size std dev | 661.073 | 680.59 | 672.11 | 629.784 | 625.649 | 596.796 | 646.396 | 611.86 | 638.084 | 560.291 | 632.263 | 
| AKAZE | # keypoints | 1351 | 1327 | 1311 | 1351 | 1360 | 1347 | 1363 | 1331 | 1357 | 1331 | 1342.9 | 
| | Time [ms] | 126.757 | 102.836 | 79.9196 | 109.864 | 96.2291 | 87.424 | 85.5551 | 81.3597 | 80.5633 | 84.7946 | 93.5302 | 
| | # selected keypoints | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 | 167 | 
| | avg. keypoint size | 7.72918 | 7.49021 | 7.45212 | 7.57523 | 7.73319 | 7.68804 | 7.73879 | 7.82613 | 7.81556 | 7.88576 | 7.69342 | 
| | keypoint size std dev | 15.3807 | 12.4225 | 12.6221 | 11.9644 | 11.8184 | 11.4138 | 11.7812 | 12.3183 | 12.2047 | 12.9699 | 12.4896 | 
| FAST | # keypoints | 5063 | 4952 | 4863 | 4840 | 4856 | 4899 | 4870 | 4868 | 4996 | 4997 | 4920.4 | 
| | Time [ms] | 2.44913 | 2.11558 | 2.33761 | 2.19016 | 2.0926 | 2.19523 | 2.22564 | 2.09203 | 2.17581 | 2.30545 | 2.21792 | 
| | # selected keypoints | 419 | 427 | 404 | 423 | 386 | 414 | 418 | 406 | 396 | 401 | 409.4 | 
| | avg. keypoint size | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 
| | keypoint size std dev | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 


Comparing the average stats:

| Detector | # keypoints | Time [ms] | # selected keypoints | avg. keypoint size | keypoint size std dev | 
|:--|:--|:--|:--|:--|:--|
| Harris | 171 | 8.42 | 24.8 | 6 | 0 |
| Shitomasi | 1342 | 11.34 | 118 | 4 | 0 |
| SIFT | 1383 | 149.2 | 138 | 5 | 35.7 |
| BRISK | 2711 | 63 | 276.2 | 22 | 213.3 |
| ORB | 500 | 18.5 | 116 | 56 | 632 |
| AKAZE | 1343 | 93.5 | 167 | 7.7 | 12.5 |
| FAST | 4920.4 | 2.2 | 409 | 7 | 0 |


### Count number of matched keypoints
Task: Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

Combinations:

| Detector | Descriptor |
|:-- |:--|
| SIFT | SIFT |
| AKAZE | AKAZE |
|SHITOMASI | BRISK |
|SHITOMASI | ORB |
|SHITOMASI | BRIEF |
|SHITOMASI | FREAK |
|HARRIS | BRISK |
|HARRIS | ORB |
|HARRIS | BRIEF |
|HARRIS | FREAK |
|FAST | BRISK |
|FAST | ORB |
|FAST | BRIEF |
|FAST | FREAK |
|BRISK | BRISK |
|BRISK | ORB |
|BRISK | BRIEF |
|BRISK | FREAK |
|ORB | BRISK |
|ORB | ORB |
|ORB | BRIEF |
|ORB | FREAK |

| Detector | Descriptor | # avg. matched keypoints | avg. detect time [ms] | avg. describe time [ms] | avg. match time [ms] | avg. total processing time [ms] |
|:--|:--|:--|:--|:--|:--|:--|
| SIFT | SIFT | 88.8889 | 135.822 | 106.77 | 0.266392 | 242.858 | 
| AKAZE | AKAZE | 139.889 | 76.9887 | 65.4199 | 0.255753 | 142.664 | 
| SHITOMASI | BRISK | 85.2222 | 10.1842 | 1.39258 | 0.190058 | 11.7669 | 
| SHITOMASI | ORB | 100.778 | 9.9952 | 3.06839 | 0.181007 | 13.2446 | 
| SHITOMASI | BRIEF | 104.889 | 9.97391 | 0.844096 | 0.197855 | 11.0159 | 
| SHITOMASI | FREAK | 85.1111 | 9.99266 | 23.6635 | 0.188879 | 33.8451 | 
| HARRIS | BRISK | 15.1111 | 6.76695 | 0.5648 | 0.0670874 | 7.39884 | 
| HARRIS | ORB | 17.4444 | 6.62285 | 2.51852 | 0.063271 | 9.20464 | 
| HARRIS | BRIEF | 18.7778 | 6.81685 | 0.399588 | 0.0590446 | 7.27548 | 
| HARRIS | FREAK | 15.6667 | 6.69206 | 22.5632 | 0.0671286 | 29.3224 | 
| FAST | BRISK | 242.556 | 2.14689 | 4.02429 | 0.911821 | 7.08299 | 
| FAST | ORB | 306.889 | 2.16057 | 3.19622 | 0.762475 | 6.11927 | 
| FAST | BRIEF | 314.556 | 2.31953 | 1.67864 | 0.766939 | 4.76511 | 
| FAST | FREAK | 247.222 | 2.29831 | 24.6845 | 0.904691 | 27.8875 | 
| BRISK | BRISK | 174.444 | 55.7211 | 2.78576 | 0.496141 | 59.003 | 
| BRISK | ORB | 167.778 | 55.8678 | 9.79248 | 0.423021 | 66.0833 | 
| BRISK | BRIEF | 189.333 | 55.6456 | 1.18702 | 0.426823 | 57.2594 | 
| BRISK | FREAK | 169.556 | 56.7937 | 25.5063 | 0.434178 | 82.7341 | 
| ORB | BRISK | 83.4444 | 16.353 | 1.33193 | 0.181018 | 17.8659 | 
| ORB | ORB | 84.5556 | 7.17155 | 9.7346 | 0.155335 | 17.0615 | 
| ORB | BRIEF | 60.5556 | 7.19071 | 0.754975 | 0.156582 | 8.10226 | 
| ORB | FREAK | 46.7778 | 7.57427 | 22.6095 | 0.13746 | 30.3212 | 

### Log runtime of algorithms
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.