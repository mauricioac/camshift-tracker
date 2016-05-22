#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <random>
#include <stack>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

// class Objeto {
//   Mat roi;

//   void track(Mat image) {

//   }
// };

void trackROI(VideoCapture * cap, Point2f tl, Point2f br, Mat roi) {

    Mat rhsv;
    rhsv = roi.clone();
    cvtColor(roi, rhsv, CV_BGR2HSV);
    Mat mask;

    cvNamedWindow("BackProjeção",CV_WINDOW_AUTOSIZE);
    cvNamedWindow( "Tracking", CV_WINDOW_AUTOSIZE );

    MatND rhist;
    int rhistsz = 180;    // bin size
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};

    calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );
    normalize(rhist,rhist,0,255,NORM_MINMAX, -1, Mat() );

    Mat frame;
    Mat fbproj;
    Mat fhsv;
    Mat fbthr;
    Rect wind;


    wind.x=tl.x;wind.y=tl.y;wind.height=br.y - tl.y;wind.width=br.x - tl.x;

    while (true) {

        cap->read(frame);
        if(frame.empty())
            return;
        cvtColor(frame,fhsv,CV_BGR2HSV);
        calcBackProject(&fhsv,1,channels,rhist,fbproj,ranges,1,true);
        normalize(fbproj,fbproj,0,255,NORM_MINMAX, -1, Mat() );
        threshold(fbproj, fbthr, 220, 1,CV_THRESH_TOZERO);
        imshow("BackPprojeção", fbthr);
        meanShift(fbthr,wind,TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
        rectangle(frame, wind, CV_RGB(255,255,255),1);
        imshow("Tracking", frame);
        waitKey(1);
    }
}

int main(int argc, char *argv[]) {

 bool stream = false;

  if (argc < 2)
  {
      stream = true;
      printf("Aviso: Faltou arquivo de video!\n");
      printf("       Usando Stream da Camera!\n");
  }

  VideoCapture cap(0);

  if(!cap.isOpened())
      return -1;

  namedWindow("aaaa", WINDOW_AUTOSIZE);
  Mat frame;

  Ptr< BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500,60,true);
  Mat mascara_background;

  mog2->setBackgroundRatio(0.001);
  Mat binaryImg;

  cap >> frame;
  mog2->apply(frame, mascara_background);

  while (true) {

    cap >> frame;
    if(frame.empty()) {
      break;
    }

    // remove sombras
    threshold(mascara_background, binaryImg, 50, 255, CV_THRESH_BINARY);
    Mat binOrig = binaryImg.clone();

    mog2->apply(frame, mascara_background);//,-0.5);

    Mat ContourImg = binaryImg.clone();

    vector< vector<Point> > contornos;
    findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < (int) contornos.size(); i++) {
      Rect bb = boundingRect(contornos[i]);
      rectangle(frame, bb, Scalar(255, 0, 128), 2);
    }

    imshow("aaaa", frame);

    if (waitKey(30) == 27) {
      break;
    }
  }

  return 0;
}
