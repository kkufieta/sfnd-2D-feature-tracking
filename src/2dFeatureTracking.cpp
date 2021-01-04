/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

void trackFeatures(string detectorType, string descriptorType,
                   bool bVis = false, bool printDetectorComparison = false,
                   bool printDetectorDescriptorComparison = false) {
  /* INIT VARIABLES AND DATA STRUCTURES */
  bool debug = false;

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix =
      "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera
                         // names have identical naming convention)
  int imgEndIndex = 9;   // last file index to load
  int imgFillWidth =
      4; // no. of digits which make up the file index (e.g. img-0001.png)

  // misc
  int dataBufferSize = 2;       // no. of images which are held in memory (ring
                                // buffer) at the same time
  vector<DataFrame> dataBuffer; // list of data frames which are held in memory
                                // at the same time

  vector<float> totalNumKeypoints;
  vector<float> detectTimes;
  vector<float> describeTimes;
  vector<float> matchTimes;
  vector<float> reducedNumKeypoints;
  vector<float> meanValues;
  vector<float> stddevValues;
  vector<float> numMatches;
  /* MAIN LOOP OVER ALL IMAGES */
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
       imgIndex++) {

    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename =
        imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file and convert to grayscale
    cv::Mat img, imgGray;
    img = cv::imread(imgFullFilename);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    if (dataBuffer.size() >= dataBufferSize) {
      dataBuffer.erase(dataBuffer.begin());
    }

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = imgGray;
    dataBuffer.push_back(frame);

    if (debug)
      cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    /* DETECT IMAGE KEYPOINTS */

    // extract 2D keypoints from current image
    vector<cv::KeyPoint>
        keypoints; // create empty feature list for current image

    // Detector types:
    // -> Gradient Based: HARRIS, SHITOMASI, SIFT
    // -> Binary: BRISK, ORB, AKAZE, FAST

    float detectTime;
    if (detectorType.compare("SHITOMASI") == 0) {
      detectTime = detKeypointsShiTomasi(keypoints, imgGray, bVis);
    } else if (detectorType.compare("HARRIS") == 0) {
      detectTime = detKeypointsHarris(keypoints, imgGray, bVis);
    } else { // SIFT, BRISK, ORB, AKAZE, FAST
      detectTime = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }
    totalNumKeypoints.push_back(keypoints.size());
    detectTimes.push_back(detectTime);

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
    if (debug) {
      cout << "After focusing on car ahead, number of keypoints: "
           << keypoints.size() << ", " << focusedKeypoints.size() << endl;
    }
    reducedNumKeypoints.push_back(keypoints.size());

    auto sum_kps = [](float a, cv::KeyPoint kp) { return a + kp.size; };
    float mean = accumulate(keypoints.begin(), keypoints.end(), 0.0, sum_kps) /
                 keypoints.size();
    auto variance_kps = [mean](float a, cv::KeyPoint kp) {
      return a + pow((kp.size - mean), 2);
    };
    float variance =
        accumulate(keypoints.begin(), keypoints.end(), 0.0, variance_kps) /
        keypoints.size();

    meanValues.push_back(mean);
    stddevValues.push_back(variance);
    if (debug) {
      cout << "Average keypoint size: " << mean << endl;
      cout << "Variance: " << variance << endl;
    }

    if (bVis) {
      cv::Mat vis_image = imgGray.clone();
      cv::drawKeypoints(img, focusedKeypoints, vis_image);
      string windowName = "Focused keypoints";
      cv::namedWindow(windowName, 6);
      cv::imshow(windowName, vis_image);
      cv::waitKey(0);
    }
    // optional : limit number of keypoints (helpful for debugging and
    // learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
      int maxKeypoints = 50;

      if (detectorType.compare("SHITOMASI") ==
          0) { // there is no response info, so keep the first 50 as they are
               // sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      if (debug)
        cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;
    if (debug)
      cout << "#2 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */
    cv::Mat descriptors;
    float describeTime = descKeypoints((dataBuffer.end() - 1)->keypoints,
                                       (dataBuffer.end() - 1)->cameraImg,
                                       descriptors, descriptorType);

    describeTimes.push_back(describeTime);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    if (debug)
      cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

    if (dataBuffer.size() >
        1) // wait until at least two images have been processed
    {

      /* MATCH KEYPOINT DESCRIPTORS */

      vector<cv::DMatch> matches;
      string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
      string descriptorType_ = descriptorType.compare("SIFT") == 0
                                   ? "DES_HOG"
                                   : "DES_BINARY"; // DES_BINARY, DES_HOG
      string selectorType = "SEL_KNN";             // SEL_NN, SEL_KNN

      float matchTime = matchDescriptors(
          (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
          (dataBuffer.end() - 2)->descriptors,
          (dataBuffer.end() - 1)->descriptors, matches, descriptorType_,
          matcherType, selectorType);

      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;
      numMatches.push_back(matches.size());
      matchTimes.push_back(matchTime);

      if (debug)
        cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

      // visualize matches between current and previous image
      if (bVis) {
        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
        cv::drawMatches((dataBuffer.end() - 2)->cameraImg,
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->cameraImg,
                        (dataBuffer.end() - 1)->keypoints, matches, matchImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        string windowName = "Matching keypoints between two camera images";
        cv::namedWindow(windowName, 7);
        cv::imshow(windowName, matchImg);
        cout << "Press key to continue to next image" << endl;
        cv::waitKey(0); // wait for key to be pressed
      }
    }

  } // eof loop over all images

  if (printDetectorComparison) {
    auto printStats = [](vector<float> stats) {
      float sum = 0;
      for (auto n : stats) {
        cout << n << " | ";
        sum += n;
      }
      cout << sum / 10 << " | \n";
    };
    cout << "| " << detectorType << " | # keypoints | ";
    printStats(totalNumKeypoints);

    cout << "| | Time [ms] | ";
    printStats(detectTimes);

    cout << "| | # selected keypoints | ";
    printStats(reducedNumKeypoints);

    cout << "| | avg. keypoint size | ";
    printStats(meanValues);

    cout << "| | keypoint size std dev | ";
    printStats(stddevValues);
  }

  if (printDetectorComparison && printDetectorDescriptorComparison)
    cout << "\n\n";

  if (printDetectorDescriptorComparison) {
    // cout << "| Detector | Descriptor | # avg. matched keypoints | avg. detect
    // time [ms] | avg. describe time [ms] | avg. match time [ms] | avg. total
    // processing time [ms] |";
    auto avgStats = [](vector<float> stats) {
      float sum = 0;
      for (float n : stats) {
        sum += n;
      }
      return sum / stats.size();
    };
    cout << "| " << detectorType << " | " << descriptorType << " | ";
    float avgDetectTimes = avgStats(detectTimes);
    float avgDescribeTimes = avgStats(describeTimes);
    float avgMatchTimes = avgStats(matchTimes);
    cout << avgStats(numMatches) << " | " << avgDetectTimes << " | "
         << avgDescribeTimes << " | " << avgMatchTimes << " | "
         << (avgDetectTimes + avgDescribeTimes + avgMatchTimes) << " | \n";
  }
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

  // Detector types:
  // -> Gradient Based: HARRIS, SHITOMASI, SIFT
  // -> Binary: BRISK, ORB, AKAZE, FAST
  string detectorType = "FAST";
  string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

  bool bVis = true; // visualize results
  bool printDetectorComparison = true;
  bool printDetectorDescriptorComparison = true;

  // SIFT works only with SIFT
  detectorType = "SIFT";
  descriptorType = "SIFT";

  trackFeatures(detectorType, descriptorType, bVis, printDetectorComparison,
                printDetectorDescriptorComparison);

  // AKAZE works only with AKAZE
  detectorType = "AKAZE";
  descriptorType = "AKAZE";

  trackFeatures(detectorType, descriptorType, bVis, printDetectorComparison,
                printDetectorDescriptorComparison);

  // Try all other combinations of detector + descriptor
  vector<string> detectors{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB"};
  vector<string> descriptors{"BRISK", "ORB", "BRIEF", "FREAK"};

  for (string detectorType : detectors) {
    for (string descriptorType : descriptors) {
      trackFeatures(detectorType, descriptorType, bVis, printDetectorComparison,
                    printDetectorDescriptorComparison);
    }
  }

  return 0;
}
