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

class Objeto {
public:

  Mat regiao;
  Mat frame;
  Mat fbproj;
  Mat fhsv;
  Mat fbthr;
  Rect wind;
  MatND rhist;
  Mat mask;
  Mat rhsv;
  int framesVivo;

  Objeto(Mat imagem, Rect r) {
    cvtColor(imagem(r), rhsv, CV_BGR2HSV);

    int rhistsz = 180;    // bin size
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};
    framesVivo = 0;

    calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );
    normalize(rhist,rhist,0,255,NORM_MINMAX, -1, Mat() );

    wind = r;
    wind.x += 8;
    wind.y += 8;
    wind.width -= 8;
    wind.height -= 8;
  }

  void track(Mat &crop, Mat &image) {
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};

    cvtColor(crop,fhsv,CV_BGR2HSV);
    // imshow("aaaa", fhsv);
    // waitKey(2000);
    calcBackProject(&fhsv,1,channels,rhist,fbproj,ranges,1,true);
    // imshow("aaaa", fbproj);
    // waitKey(2000);
    threshold(fbproj, fbthr, 220, 1,CV_THRESH_TOZERO);
    normalize(fbthr,fbthr,0,255,NORM_MINMAX, -1, Mat() );
    // imshow("aaaa", fbthr);
    // waitKey(2000);
    RotatedRect rect = CamShift(fbthr,wind,TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 0.1));
    wind = rect.boundingRect();

    Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
      line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
    }

    framesVivo++;
  }
};

Mat img, imgtemp;
Rect roi;
Point p1, p2, p3;
 bool
  flag1 = false,
  flag2 = false,
  flag3 = false,
  flag4 = true;

void on_mouse( int event, int x, int y, int flags, void* param )
{

  if (!flag3)
    return;

  switch(event)
  {
    case CV_EVENT_LBUTTONDOWN:
        if (!flag2) {
          p3.x = x;
          p3.y = y;
          flag1 = true;
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        imgtemp = img.clone();
        if (flag1) {
                    rectangle(imgtemp,p3, Point(x,y+5),  CV_RGB(255,255,255),1);
        }
        imshow( "Imagem", imgtemp);
        break;
    case CV_EVENT_LBUTTONUP:
        p1.x = p3.x;
        p1.y = p3.y;
        p2.x = x;
        p2.y = y;
        flag1 = false;
        flag2 = true;
        break;
    default:
      break;
  }
}

bool pegaROI() {

  imgtemp = img.clone();

  if(img.empty())
     return -1;

  flag1 = flag2 = false;
  flag3 = true;

  p1.x = p1.y = 10;
  p2.x = img.cols-10;
  p2.y = img.rows-10;


  cvNamedWindow( "Imagem", CV_WINDOW_AUTOSIZE );
  setMouseCallback("Imagem", on_mouse, 0 );
  imshow( "Imagem", img);

  if (img.rows< 50 || img.cols < 50) {
      printf("Erro: imagem ḿuito pequena!\n");
      return -1;
  }
  imgtemp = img.clone();

  while(!flag2) {
    if(waitKey(10) == 27) {
      break;
    }
    if (flag2) {
      break;
    }
  }

  cvDestroyWindow("Imagem");
    roi = Rect(p1,p2);
    return true;
}

int main(int argc, char *argv[]) {

 bool stream = false;

  if (argc < 2)
  {
      stream = true;
  }

  VideoCapture *cap;

  if (stream) {
    cap = new VideoCapture(0);
  } else {
    cap = new VideoCapture(argv[1]);
  }

  if(!cap->isOpened())
      return -1;

  Mat frame;

  Ptr< BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500,60,true);
  Mat mascara_background;

  mog2->setBackgroundRatio(0.001);
  Mat binaryImg;

  for (int i = 0; i < 80; i++) {
    *cap >> frame;
    mog2->apply(frame, mascara_background);
  }

  img = frame.clone();
  namedWindow("aaaa", WINDOW_AUTOSIZE);

  Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 7), Point(1,3) );

  pegaROI();

  int contador = 0;

  vector<Objeto> objetos;

  while (true) {

    *cap >> frame;
    if(frame.empty()) {
      break;
    }

    // remove sombras
    threshold(mascara_background, binaryImg, 50, 255, CV_THRESH_BINARY);

    // for (int i = 0; i < 10; i++)
    //   morphologyEx(binaryImg, binaryImg, CV_MOP_CLOSE, elemento);

    Mat binOrig = binaryImg.clone();

    mog2->apply(frame, mascara_background);//,-0.5);

    Mat ContourImg = binaryImg.clone();

    vector< vector<Point> > contornos;
    findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    Mat mask = Mat::zeros(ContourImg.rows, ContourImg.cols, CV_8UC3);
    drawContours(mask, contornos, -1, Scalar(255), CV_FILLED);

    Mat mask2 = Mat::zeros(ContourImg.rows, ContourImg.cols, CV_8UC3);
    drawContours(mask2, contornos, -1, Scalar(255), CV_FILLED);

    Mat crop(frame.rows, frame.cols, frame.type());
    crop.setTo(Scalar(255,255,255));
    frame.copyTo(crop, mask);

    Mat crop2(frame.rows, frame.cols, frame.type());
    frame.copyTo(crop2, mask2);

    for (int i = 0; i < (int) contornos.size(); i++) {
      Rect bb = boundingRect(contornos[i]);

      if (bb.width <= 10 || bb.height <= 10)
        continue;

      bool achou = false;
      for (int j = 0; j < objetos.size(); j++) {
        Rect intersecao = bb & objetos[j].wind;

        if (intersecao.height > 7 || intersecao.width > 10) {
          achou = true;
        }
      }

      if (achou) {
        continue;
      }

        // rectangle(frame, bb, Scalar(255, 0, 128), 2);
      if (bb.y >= roi.y && bb.y <= roi.y + 10) {
        // cout << "novo?" << endl;
        // imshow("aaaa", crop);
        // waitKey(2000);
        Objeto o(crop, bb);
        objetos.push_back(o);
      }
    }

    // rectangle(frame, roi, Scalar(255, 0, 0), 2);

    vector<Objeto> novos_objetos;

    if (objetos.size() > 0) {

      // imshow("aaaa", crop);
      // waitKey(4000);
    }

    for (int i = 0; i < (int) objetos.size(); i++) {
      objetos[i].track(crop, frame);

      if (objetos[i].wind.y > roi.y + roi.height + 5) {
        contador += 1;
        continue;
      }

      if (objetos[i].wind.height > 90 || objetos[i].wind.width > 120)
      {
        continue;
      }

      novos_objetos.push_back(objetos[i]);
    }

    objetos = novos_objetos;

    string text = "Contador: " + to_string(contador);
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    cv::Point textOrg(10, 30);

    putText(frame, text, textOrg, fontFace, fontScale, Scalar(0, 0, 255), thickness,8);

    imshow("aaaa", frame);

    if (waitKey(30) == 27) {
      break;
    }
  }

  return 0;
}
