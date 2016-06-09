/**
 * Alunos: Silvana Trindade e Maurício André Cinelli
 * Trabalho IV: tracking e contagem de objetos com CamShift
 * Opencv versão 3.0
 */

#include <iostream>
#include <opencv2/core/core.hpp>
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

  Objeto(Mat imagem, Rect r) {
    cvtColor(imagem(r), rhsv, CV_BGR2HSV);

    int rhistsz = 180;    // bin size
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};

    calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );
    normalize(rhist,rhist,0,255,NORM_MINMAX, -1, Mat() );

    wind = r;
  }

  int track(Mat &crop, Mat &image) {
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};

    try {
	    // converte pra HSV
	    cvtColor(crop,fhsv,CV_BGR2HSV);
	    // calcula back projeção
	    calcBackProject(&fhsv,1,channels,rhist,fbproj,ranges,1,true);

	    // elimina valores que não batem com o objeto
	    threshold(fbproj, fbthr, 200, 1,CV_THRESH_TOZERO);
	    normalize(fbthr,fbthr,0,255,NORM_MINMAX, -1, Mat() );

	    // Executa camshift
	    // 5 iterações, precisão 0.1
	    RotatedRect rect = CamShift(fbthr,wind,TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 0.1));

	    // desenha retangulo de tracking na imagem
	    Point2f vertices[4];
	    rect.points(vertices);
	    for (int i = 0; i < 4; i++) {
	      line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
	    }

      return 1;
    } catch (int e) {
      return -1;
    }
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

 // verifica se vai pegar video de arquivo ou câmera
 bool stream = false;

  if (argc < 2)
  {
      stream = true;
  }

  VideoCapture *cap;

  if (stream) {
    cap = new VideoCapture(1);
  } else {
    cap = new VideoCapture(argv[1]);
  }

  if(!cap->isOpened())
      return -1;

  Mat frame;

  // Cria subtração de background por MOG2
  Ptr< BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500,60,true);
  Mat mascara_background;
  Mat binaryImg;

  mog2->setBackgroundRatio(0.01);

  // pega X frames para estabilizar background
  for (int i = 0; i < 80; i++) {
    *cap >> frame;
    mog2->apply(frame, mascara_background);
  }

  img = frame.clone();
  namedWindow("video", WINDOW_AUTOSIZE);

  // matriz de morfologia
  Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 1), Point(1,0) );

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

    Mat binOrig = binaryImg.clone();

    // aplica frame ao background
    mog2->apply(frame, mascara_background);

    Mat ContourImg = binaryImg.clone();

    // pega contornos
    vector< vector<Point> > contornos;
    findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Cria uma imagem preta, pintando os contornos de branco
    Mat mask = Mat::zeros(ContourImg.rows, ContourImg.cols, CV_8UC3);
    drawContours(mask, contornos, -1, Scalar(255), CV_FILLED);

    // cria uma nova imagem, sobrepondo o frame com a máscara
    // assim pinta somente os objetos, num fundo branco
    Mat crop(frame.rows, frame.cols, frame.type());
    crop.setTo(Scalar(255,255,255));
    frame.copyTo(crop, mask);


    for (int i = 0; i < (int) contornos.size(); i++) {
      Rect bb = boundingRect(contornos[i]);

      // objeto muito pequeno, cai fora
      if (bb.width <= 10 || bb.height <= 10)
        continue;

      // verifica se objeto já foi detectado em um frame anterior
      // por exemplo se a leitura e muito rapida
      bool achou = false;
      for (int j = 0; j < (int) objetos.size(); j++) {
        Rect intersecao = bb & objetos[j].wind;

        if (intersecao.height > 8 || intersecao.width > 5) {
          achou = true;
        }
      }

      if (achou) {
        continue;
      }

      // se não encontrar, cria um novo objeto
      if (bb.y >= roi.y && bb.y <= roi.y + 10) {
        Objeto o(crop, bb);
        objetos.push_back(o);
      }
    }

    // percorre objetos, chamando Camshift
    vector<Objeto> novos_objetos;
    for (int i = 0; i < (int) objetos.size(); i++) {
      int result = objetos[i].track(crop, frame);

      if (result == -1) continue;

      // se o objeto passou da parte de baixo do quadrado
      // conta o objeto e remove-o
      if (objetos[i].wind.y > roi.y + roi.height + 5) {
        contador += 1;
        continue;
      }

      // se a janela do camshift ficar muito grande, remove objeto
      if (objetos[i].wind.height > 90 || objetos[i].wind.width > 120)
      {
        continue;
      }

      // se objeto deve continuar, coloca-o em um novo vetor
      novos_objetos.push_back(objetos[i]);
    }

    objetos = novos_objetos;

    // mostra contagem na tela
    string text = "Contador: " + to_string(contador);
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    cv::Point textOrg(10, 30);

    putText(frame, text, textOrg, fontFace, fontScale, Scalar(0, 0, 255), thickness,8);

    imshow("video", frame);

    if (waitKey(10) == 27) {
      break;
    }
  }

  cout << "Contou " << contador << " objetos no total" << endl;

  return 0;
}
